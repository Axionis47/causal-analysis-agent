"""E2E Test CLI for Causal Analysis API.

This script acts as a real HTTP client against the localhost API, testing the
entire stack from API → Celery → Agents → Database → Results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.ground_truth import GroundTruth

logger = logging.getLogger(__name__)

# Artifacts directory for failure diagnostics
ARTIFACTS_DIR = Path(__file__).parent.parent / "evals" / "artifacts"


class E2ETestClient:
    """HTTP client wrapper for E2E testing."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Initialize E2E test client.

        Args:
            base_url: Base URL for the API.
        """
        self.base_url = base_url.rstrip("/")
        self._token: str | None = None
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "E2ETestClient":
        """Enter async context."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0, read=300.0),
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context."""
        if self._client:
            await self._client.aclose()

    def set_token(self, token: str) -> None:
        """Set authentication token.

        Args:
            token: JWT access token.
        """
        self._token = token

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication.

        Returns:
            Dictionary of headers.
        """
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def get(self, path: str) -> httpx.Response:
        """Make GET request.

        Args:
            path: API path.

        Returns:
            HTTP response.

        Raises:
            httpx.HTTPStatusError: On HTTP error.
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        response = await self._client.get(path, headers=self._get_headers())
        response.raise_for_status()
        return response

    async def post(self, path: str, json: dict[str, Any] | None = None) -> httpx.Response:
        """Make POST request.

        Args:
            path: API path.
            json: JSON request body.

        Returns:
            HTTP response.

        Raises:
            httpx.HTTPStatusError: On HTTP error.
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        response = await self._client.post(path, json=json, headers=self._get_headers())
        response.raise_for_status()
        return response


class AnalysisValidator:
    """Validates analysis results against ground truth."""

    def __init__(self) -> None:
        """Initialize validator with ground truth."""
        self._gt = GroundTruth()

    def validate_ate(self, dataset_id: str, estimated_ate: float) -> dict[str, Any]:
        """Validate ATE against ground truth.

        Args:
            dataset_id: Dataset identifier.
            estimated_ate: Estimated ATE value.

        Returns:
            Validation result dictionary.
        """
        return self._gt.validate_ate(dataset_id, estimated_ate)

    def validate_graph(
        self, dataset_id: str, predicted_edges: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Validate graph edges against ground truth.

        Args:
            dataset_id: Dataset identifier.
            predicted_edges: List of predicted edges.

        Returns:
            Validation result dictionary.
        """
        return self._gt.validate_graph(dataset_id, predicted_edges)

    def validate_refutation(self, dataset_id: str, pass_rate: float) -> dict[str, Any]:
        """Validate refutation pass rate.

        Args:
            dataset_id: Dataset identifier.
            pass_rate: Refutation test pass rate.

        Returns:
            Validation result dictionary.
        """
        return self._gt.validate_refutation(dataset_id, pass_rate)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for E2E tests.

    Args:
        verbose: Enable verbose (DEBUG) logging.
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Create log directory if needed
    log_dir = Path(__file__).parent.parent / "evals"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "e2e_test.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )

    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def retry_request(
    func: Any,
    max_attempts: int = 3,
    backoff: float = 2.0,
) -> Any:
    """Retry HTTP requests with exponential backoff.

    Args:
        func: Async function to retry.
        max_attempts: Maximum retry attempts.
        backoff: Backoff multiplier.

    Returns:
        Function result.

    Raises:
        httpx.HTTPStatusError: On non-retryable HTTP error.
    """
    for attempt in range(max_attempts):
        try:
            return await func()
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:
                raise  # Don't retry client errors
            if attempt == max_attempts - 1:
                raise
            wait_time = backoff**attempt
            logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)


async def setup_test_user(client: E2ETestClient) -> str:
    """Register test user and return access token.

    Args:
        client: E2E test client.

    Returns:
        Access token.
    """
    email = f"e2e_test_{uuid.uuid4().hex[:8]}@test.local"
    password = "TestPassword123!"

    logger.info(f"Registering test user: {email}")

    # POST /api/v1/auth/register
    try:
        await client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password},
        )
    except httpx.HTTPStatusError as e:
        # User may already exist, continue to login
        if e.response.status_code != 409:
            raise
        logger.warning(f"User {email} may already exist, attempting login...")

    # POST /api/v1/auth/token
    response = await client.post(
        "/api/v1/auth/token",
        json={"email": email, "password": password},
    )
    data = response.json()
    token = data["access_token"]
    client.set_token(token)

    logger.info(f"Test user authenticated: {email}")
    return token


async def submit_analysis(client: E2ETestClient, dataset_id: str) -> dict[str, Any]:
    """Submit analysis for a dataset and return analysis metadata.

    Args:
        client: E2E test client.
        dataset_id: Dataset identifier.

    Returns:
        Analysis metadata dictionary.
    """
    gt = GroundTruth()
    kaggle_info = gt.get_kaggle_info(dataset_id)
    analysis_config = gt.get_analysis_config(dataset_id)

    # POST /api/v1/analyses
    response = await client.post(
        "/api/v1/analyses",
        json={
            "kaggle_url": kaggle_info["url"],
            "config": analysis_config,
        },
    )

    analysis = response.json()
    logger.info(f"Analysis created: {analysis['id']} for {dataset_id}")

    return {
        "analysis_id": analysis["id"],
        "dataset_id": dataset_id,
        "status": analysis.get("status", "pending"),
        "created_at": analysis.get("created_at"),
    }


async def monitor_progress(
    client: E2ETestClient,
    analysis_id: str,
    timeout: int,
) -> dict[str, Any]:
    """Stream progress via SSE and return final status.

    Args:
        client: E2E test client.
        analysis_id: Analysis identifier.
        timeout: Timeout in seconds.

    Returns:
        Progress result dictionary with:
            - status: 'completed', 'failed', 'timeout', 'error', or 'unknown'
            - sse_transcript: List of all received SSE events for debugging
            - last_event: Last progress event data
            - final_event: Final event data (if completed)
            - error: Error message (if failed)
    """
    url = f"{client.base_url}/api/v1/analyses/{analysis_id}/stream"
    headers = {"Authorization": f"Bearer {client._token}"}

    start_time = time.time()
    last_event: dict[str, Any] | None = None
    current_event_name: str | None = None
    sse_transcript: list[dict[str, Any]] = []

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout + 60.0)) as http_client:
            async with http_client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if time.time() - start_time > timeout:
                        logger.error(f"Analysis {analysis_id} exceeded {timeout}s timeout")
                        return {"status": "timeout", "last_event": last_event, "sse_transcript": sse_transcript}

                    # Skip empty lines (SSE event separator)
                    if not line:
                        current_event_name = None
                        continue

                    # Parse SSE event: lines to capture the event name
                    if line.startswith("event:"):
                        current_event_name = line[6:].strip()
                        continue

                    # Parse SSE data: lines
                    if not line.startswith("data:"):
                        # Ignore comment lines (heartbeats) and other fields
                        continue

                    try:
                        data_str = line[5:].strip()  # Remove "data:" prefix
                        if not data_str:
                            continue
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Record SSE event for transcript
                    sse_transcript.append({
                        "event": current_event_name,
                        "data": data,
                        "timestamp": time.time() - start_time,
                    })

                    # Handle 'disconnected' event as terminal status
                    if current_event_name == "disconnected":
                        reason = data.get("reason", "unknown")
                        logger.info(f"[{analysis_id[:8]}...] Stream disconnected: {reason}")

                        if reason == "completed":
                            return {
                                "status": "completed",
                                "final_event": data,
                                "sse_transcript": sse_transcript,
                            }
                        elif reason == "timeout":
                            return {
                                "status": "timeout",
                                "last_event": last_event,
                                "sse_transcript": sse_transcript,
                            }
                        elif reason == "error":
                            return {
                                "status": "failed",
                                "error": data.get("message", "Stream error"),
                                "last_event": last_event,
                                "sse_transcript": sse_transcript,
                            }
                        else:
                            # stream_end or unknown reason - need fallback polling
                            return {
                                "status": "unknown",
                                "last_event": last_event,
                                "sse_transcript": sse_transcript,
                            }

                    # Handle 'progress' event using progress_percent and status fields
                    if current_event_name == "progress":
                        stage = data.get("stage", "unknown")
                        progress_percent = data.get("progress_percent", 0)
                        status = data.get("status", "unknown")
                        message = data.get("message", "")

                        logger.info(f"[{analysis_id[:8]}...] {stage}: {progress_percent}% - {message}")
                        last_event = data

                        # Check if status indicates terminal state
                        if status == "completed":
                            logger.info(f"[{analysis_id[:8]}...] Analysis completed successfully")
                            return {
                                "status": "completed",
                                "final_event": data,
                                "sse_transcript": sse_transcript,
                            }
                        elif status in ("failed", "error"):
                            error_msg = data.get("error") or data.get("message", "Unknown error")
                            logger.error(f"[{analysis_id[:8]}...] Analysis failed: {error_msg}")
                            return {
                                "status": "failed",
                                "error": error_msg,
                                "last_event": last_event,
                                "sse_transcript": sse_transcript,
                            }

    except httpx.TimeoutException:
        logger.error(f"Analysis {analysis_id} connection timed out")
        return {"status": "timeout", "last_event": last_event, "sse_transcript": sse_transcript}
    except httpx.HTTPStatusError as e:
        logger.error(f"Analysis {analysis_id} stream error: {e}")
        return {"status": "error", "error": str(e), "sse_transcript": sse_transcript}

    return {"status": "unknown", "last_event": last_event, "sse_transcript": sse_transcript}


async def poll_status(
    client: E2ETestClient,
    analysis_id: str,
    timeout: int,
    poll_interval: int = 5,
) -> dict[str, Any]:
    """Poll analysis status as fallback when SSE fails.

    Args:
        client: E2E test client.
        analysis_id: Analysis identifier.
        timeout: Timeout in seconds.
        poll_interval: Polling interval in seconds.

    Returns:
        Status result dictionary.
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = await client.get(f"/api/v1/analyses/{analysis_id}")
            data = response.json()
            status = data.get("status", "unknown")

            logger.info(f"[{analysis_id[:8]}...] Status: {status}")

            if status == "completed":
                return {"status": "completed", "data": data}
            elif status in ("failed", "error"):
                return {"status": "failed", "error": data.get("error", "Unknown error")}

            await asyncio.sleep(poll_interval)

        except httpx.HTTPStatusError as e:
            logger.warning(f"Status poll failed: {e}")
            await asyncio.sleep(poll_interval)

    return {"status": "timeout"}


async def fetch_results(client: E2ETestClient, analysis_id: str) -> dict[str, Any]:
    """Fetch all analysis results from API.

    Args:
        client: E2E test client.
        analysis_id: Analysis identifier.

    Returns:
        Results dictionary with parsed metrics.
    """
    results: dict[str, Any] = {
        "analysis_id": analysis_id,
        "ate": None,
        "ate_ci_lower": None,
        "ate_ci_upper": None,
        "p_value": None,
        "graph_edges": [],
        "graph_nodes": [],
        "refutation_pass_rate": None,
        "overall_confidence": None,
    }

    # Try export endpoint first
    try:
        response = await client.get(f"/api/v1/results/{analysis_id}/export")
        export_data = response.json()

        results["summary"] = export_data.get("summary", {})
        results["technical"] = export_data.get("technical", {})
        results["visualizations"] = export_data.get("visualizations", {})
        results["validation"] = export_data.get("validation", {})

        # Parse treatment effects
        treatment_effects = results["technical"].get("treatment_effects", [])
        if treatment_effects:
            first_effect = treatment_effects[0]
            results["ate"] = first_effect.get("ate")
            results["ate_ci_lower"] = first_effect.get("ci_lower")
            results["ate_ci_upper"] = first_effect.get("ci_upper")
            results["p_value"] = first_effect.get("p_value")

        # Parse causal graph
        causal_graph = results["visualizations"].get("causal_graph", {})
        results["graph_edges"] = causal_graph.get("edges", [])
        results["graph_nodes"] = causal_graph.get("nodes", [])

        # Parse validation results
        validation_data = results["validation"]
        refutation_tests = validation_data.get("refutation_tests", [])
        if refutation_tests:
            passed_tests = [t for t in refutation_tests if t.get("passed", False)]
            results["refutation_pass_rate"] = len(passed_tests) / len(refutation_tests)
        results["overall_confidence"] = validation_data.get("overall_confidence", 0.0)

        return results

    except httpx.HTTPStatusError as e:
        logger.warning(f"Export endpoint failed ({e}), trying individual endpoints...")

    # Fallback: fetch from individual endpoints
    try:
        response = await client.get(f"/api/v1/results/{analysis_id}/technical")
        technical_data = response.json()
        treatment_effects = technical_data.get("treatment_effects", [])
        if treatment_effects:
            first_effect = treatment_effects[0]
            results["ate"] = first_effect.get("ate")
            results["ate_ci_lower"] = first_effect.get("ci_lower")
            results["ate_ci_upper"] = first_effect.get("ci_upper")
            results["p_value"] = first_effect.get("p_value")
    except httpx.HTTPStatusError:
        logger.warning("Technical endpoint failed")

    try:
        response = await client.get(f"/api/v1/results/{analysis_id}/visualizations")
        viz_data = response.json()
        causal_graph = viz_data.get("causal_graph", {})
        results["graph_edges"] = causal_graph.get("edges", [])
        results["graph_nodes"] = causal_graph.get("nodes", [])
    except httpx.HTTPStatusError:
        logger.warning("Visualizations endpoint failed")

    try:
        response = await client.get(f"/api/v1/results/{analysis_id}/validation")
        validation_data = response.json()
        refutation_tests = validation_data.get("refutation_tests", [])
        if refutation_tests:
            passed_tests = [t for t in refutation_tests if t.get("passed", False)]
            results["refutation_pass_rate"] = len(passed_tests) / len(refutation_tests)
        results["overall_confidence"] = validation_data.get("overall_confidence", 0.0)
    except httpx.HTTPStatusError:
        logger.warning("Validation endpoint failed")

    return results


async def validate_results(dataset_id: str, results: dict[str, Any]) -> dict[str, Any]:
    """Validate analysis results against ground truth.

    Args:
        dataset_id: Dataset identifier.
        results: Analysis results dictionary.

    Returns:
        Validation report dictionary.
    """
    gt = GroundTruth()

    validation_report: dict[str, Any] = {
        "dataset_id": dataset_id,
        "analysis_id": results["analysis_id"],
        "timestamp": datetime.utcnow().isoformat(),
        "validations": {},
    }

    # Validate ATE
    if results.get("ate") is not None:
        try:
            ate_validation = gt.validate_ate(dataset_id, results["ate"])
            validation_report["validations"]["ate"] = ate_validation
            status = "PASS" if ate_validation["passed"] else "FAIL"
            logger.info(f"[{dataset_id}] ATE Validation: {status}")
            logger.info(
                f"  Estimated: {results['ate']:.3f}, "
                f"Expected: {ate_validation['expected_range']}"
            )
        except ValueError as e:
            logger.warning(f"[{dataset_id}] ATE validation skipped: {e}")
            validation_report["validations"]["ate"] = {"passed": None, "error": str(e)}

    # Validate Graph
    if results.get("graph_edges"):
        graph_validation = gt.validate_graph(dataset_id, results["graph_edges"])
        validation_report["validations"]["graph"] = graph_validation
        status = "PASS" if graph_validation["passed"] else "FAIL"
        logger.info(f"[{dataset_id}] Graph Validation: {status}")
        logger.info(
            f"  F1: {graph_validation['f1']:.3f}, Threshold: {graph_validation['threshold']}"
        )

    # Validate Refutation
    if results.get("refutation_pass_rate") is not None:
        refutation_validation = gt.validate_refutation(dataset_id, results["refutation_pass_rate"])
        validation_report["validations"]["refutation"] = refutation_validation
        status = "PASS" if refutation_validation["passed"] else "FAIL"
        logger.info(f"[{dataset_id}] Refutation Validation: {status}")
        logger.info(
            f"  Pass Rate: {results['refutation_pass_rate']:.3f}, "
            f"Threshold: {refutation_validation['threshold']}"
        )

    # Overall pass/fail
    validations = validation_report["validations"].values()
    passed_validations = [v for v in validations if v.get("passed") is not None]
    all_passed = all(v.get("passed", False) for v in passed_validations) if passed_validations else False
    validation_report["overall_passed"] = all_passed
    validation_report["overall_status"] = "PASS" if all_passed else "FAIL"

    return validation_report


async def save_failure_artifacts(
    client: E2ETestClient,
    dataset_id: str,
    analysis_id: str | None,
    error: str | None,
    sse_transcript: list[dict[str, Any]] | None = None,
    http_trace: dict[str, Any] | None = None,
) -> Path | None:
    """Save failure artifacts for debugging.

    Creates a timestamped directory under backend/evals/artifacts/ containing:
    - trace.json: SSE transcript and HTTP request/response bodies
    - report.html: HTML report downloaded from the API (if available)
    - screenshot.png: Screenshot of the HTML report (if playwright is available)

    Args:
        client: E2E test client.
        dataset_id: Dataset identifier.
        analysis_id: Analysis identifier (if available).
        error: Error message.
        sse_transcript: List of SSE events received.
        http_trace: HTTP request/response trace data.

    Returns:
        Path to the artifacts directory, or None if saving failed.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    artifact_dir = ARTIFACTS_DIR / f"{dataset_id}_{timestamp}"

    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Save trace file with SSE transcript and HTTP trace
        trace_data = {
            "dataset_id": dataset_id,
            "analysis_id": analysis_id,
            "timestamp": datetime.utcnow().isoformat(),
            "error": error,
            "sse_transcript": sse_transcript or [],
            "http_trace": http_trace or {},
        }
        trace_file = artifact_dir / "trace.json"
        trace_file.write_text(json.dumps(trace_data, indent=2, default=str))
        logger.info(f"Saved trace file: {trace_file}")

        # Try to download HTML report if we have an analysis_id
        if analysis_id:
            html_file = await download_html_report(client, analysis_id, artifact_dir)
            if html_file:
                # Try to generate screenshot from HTML
                await generate_screenshot_from_html(html_file, artifact_dir)

        logger.info(f"Failure artifacts saved to: {artifact_dir}")
        return artifact_dir

    except Exception as e:
        logger.error(f"Failed to save failure artifacts: {e}")
        return None


async def download_html_report(
    client: E2ETestClient,
    analysis_id: str,
    artifact_dir: Path,
) -> Path | None:
    """Download HTML report for an analysis.

    Args:
        client: E2E test client.
        analysis_id: Analysis identifier.
        artifact_dir: Directory to save the report.

    Returns:
        Path to the downloaded HTML file, or None if download failed.
    """
    try:
        url = f"/api/v1/results/{analysis_id}/download"
        params = {"format": "html"}

        if not client._client:
            raise RuntimeError("Client not initialized")

        response = await client._client.get(
            url,
            params=params,
            headers=client._get_headers(),
            follow_redirects=True,
        )

        if response.status_code == 200:
            html_file = artifact_dir / "report.html"
            html_file.write_bytes(response.content)
            logger.info(f"Downloaded HTML report: {html_file}")
            return html_file
        elif response.status_code == 307:
            # Handle redirect to signed URL
            redirect_url = response.headers.get("location")
            if redirect_url:
                async with httpx.AsyncClient() as redirect_client:
                    redirect_response = await redirect_client.get(redirect_url)
                    if redirect_response.status_code == 200:
                        html_file = artifact_dir / "report.html"
                        html_file.write_bytes(redirect_response.content)
                        logger.info(f"Downloaded HTML report from redirect: {html_file}")
                        return html_file
        else:
            logger.warning(f"HTML report download failed with status {response.status_code}")

    except Exception as e:
        logger.warning(f"Failed to download HTML report: {e}")

    return None


async def generate_screenshot_from_html(html_file: Path, artifact_dir: Path) -> Path | None:
    """Generate a screenshot from an HTML file using playwright.

    Args:
        html_file: Path to the HTML file.
        artifact_dir: Directory to save the screenshot.

    Returns:
        Path to the screenshot file, or None if generation failed.
    """
    try:
        # Try to import playwright - this is optional
        from playwright.async_api import async_playwright
    except ImportError:
        logger.warning("Playwright not installed - skipping screenshot generation. "
                      "Install with: pip install playwright && playwright install chromium")
        return None

    screenshot_file = artifact_dir / "screenshot.png"

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()

            # Load the HTML file
            html_content = html_file.read_text()
            await page.set_content(html_content)

            # Wait for any rendering to complete
            await page.wait_for_load_state("networkidle")

            # Take full-page screenshot
            await page.screenshot(path=str(screenshot_file), full_page=True)
            await browser.close()

            logger.info(f"Generated screenshot: {screenshot_file}")
            return screenshot_file

    except Exception as e:
        logger.warning(f"Failed to generate screenshot: {e}")
        return None


async def run_single_dataset(
    client: E2ETestClient,
    dataset_id: str,
    timeout: int,
) -> list[dict[str, Any]]:
    """Run E2E test for a single dataset.

    Args:
        client: E2E test client.
        dataset_id: Dataset identifier.
        timeout: Analysis timeout in seconds.

    Returns:
        List containing single result dictionary.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting E2E test for dataset: {dataset_id}")
    logger.info(f"{'='*60}\n")

    analysis_id: str | None = None
    sse_transcript: list[dict[str, Any]] = []

    try:
        # Submit analysis
        analysis_meta = await submit_analysis(client, dataset_id)
        analysis_id = analysis_meta["analysis_id"]

        # Monitor progress via SSE
        progress_result = await monitor_progress(client, analysis_id, timeout)
        sse_transcript = progress_result.get("sse_transcript", [])

        # Fallback to polling if SSE failed
        if progress_result["status"] not in ("completed", "failed"):
            logger.info("SSE stream ended, falling back to polling...")
            progress_result = await poll_status(client, analysis_id, timeout)

        if progress_result["status"] != "completed":
            error_msg = progress_result.get("error", "Timeout or unknown error")

            # Save failure artifacts
            await save_failure_artifacts(
                client=client,
                dataset_id=dataset_id,
                analysis_id=analysis_id,
                error=error_msg,
                sse_transcript=sse_transcript,
                http_trace={"progress_result": progress_result},
            )

            return [
                {
                    "dataset_id": dataset_id,
                    "analysis_id": analysis_id,
                    "status": "failed",
                    "error": error_msg,
                }
            ]

        # Fetch results
        analysis_results = await fetch_results(client, analysis_id)

        # Validate against ground truth
        validation_report = await validate_results(dataset_id, analysis_results)

        # Check if validation failed - also save artifacts for validation failures
        if not validation_report.get("overall_passed", False):
            await save_failure_artifacts(
                client=client,
                dataset_id=dataset_id,
                analysis_id=analysis_id,
                error="Validation failed",
                sse_transcript=sse_transcript,
                http_trace={
                    "progress_result": progress_result,
                    "validation_report": validation_report,
                },
            )

        return [
            {
                "dataset_id": dataset_id,
                "analysis_id": analysis_id,
                "status": "completed",
                "validation": validation_report,
                "results": analysis_results,
            }
        ]

    except Exception as e:
        logger.error(f"[{dataset_id}] Test failed with exception: {e}", exc_info=True)

        # Save failure artifacts for exceptions
        await save_failure_artifacts(
            client=client,
            dataset_id=dataset_id,
            analysis_id=analysis_id,
            error=str(e),
            sse_transcript=sse_transcript,
            http_trace={"exception_type": type(e).__name__},
        )

        return [
            {
                "dataset_id": dataset_id,
                "analysis_id": analysis_id,
                "status": "error",
                "error": str(e),
            }
        ]


async def run_all_datasets(client: E2ETestClient, timeout: int) -> list[dict[str, Any]]:
    """Run E2E tests for all datasets sequentially.

    Args:
        client: E2E test client.
        timeout: Analysis timeout in seconds.

    Returns:
        List of result dictionaries for all datasets.
    """
    gt = GroundTruth()
    dataset_ids = gt.dataset_ids

    results: list[dict[str, Any]] = []

    for dataset_id in dataset_ids:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting E2E test for dataset: {dataset_id}")
        logger.info(f"{'='*60}\n")

        analysis_id: str | None = None
        sse_transcript: list[dict[str, Any]] = []

        try:
            # Submit analysis
            analysis_meta = await submit_analysis(client, dataset_id)
            analysis_id = analysis_meta["analysis_id"]

            # Monitor progress via SSE
            progress_result = await monitor_progress(client, analysis_id, timeout)
            sse_transcript = progress_result.get("sse_transcript", [])

            # Fallback to polling if SSE failed
            if progress_result["status"] not in ("completed", "failed"):
                logger.info("SSE stream ended, falling back to polling...")
                progress_result = await poll_status(client, analysis_id, timeout)

            if progress_result["status"] != "completed":
                error_msg = progress_result.get("error", "Timeout or unknown error")

                # Save failure artifacts
                await save_failure_artifacts(
                    client=client,
                    dataset_id=dataset_id,
                    analysis_id=analysis_id,
                    error=error_msg,
                    sse_transcript=sse_transcript,
                    http_trace={"progress_result": progress_result},
                )

                results.append(
                    {
                        "dataset_id": dataset_id,
                        "analysis_id": analysis_id,
                        "status": "failed",
                        "error": error_msg,
                    }
                )
                continue

            # Fetch results
            analysis_results = await fetch_results(client, analysis_id)

            # Validate against ground truth
            validation_report = await validate_results(dataset_id, analysis_results)

            # Check if validation failed - also save artifacts for validation failures
            if not validation_report.get("overall_passed", False):
                await save_failure_artifacts(
                    client=client,
                    dataset_id=dataset_id,
                    analysis_id=analysis_id,
                    error="Validation failed",
                    sse_transcript=sse_transcript,
                    http_trace={
                        "progress_result": progress_result,
                        "validation_report": validation_report,
                    },
                )

            results.append(
                {
                    "dataset_id": dataset_id,
                    "analysis_id": analysis_id,
                    "status": "completed",
                    "validation": validation_report,
                    "results": analysis_results,
                }
            )

        except Exception as e:
            logger.error(f"[{dataset_id}] Test failed with exception: {e}", exc_info=True)

            # Save failure artifacts for exceptions
            await save_failure_artifacts(
                client=client,
                dataset_id=dataset_id,
                analysis_id=analysis_id,
                error=str(e),
                sse_transcript=sse_transcript,
                http_trace={"exception_type": type(e).__name__},
            )

            results.append(
                {
                    "dataset_id": dataset_id,
                    "analysis_id": analysis_id,
                    "status": "error",
                    "error": str(e),
                }
            )

    return results


def generate_report(results: list[dict[str, Any]], output_path: str | None = None) -> dict[str, Any]:
    """Generate JSON report with summary statistics.

    Args:
        results: List of test result dictionaries.
        output_path: Optional path to save JSON report.

    Returns:
        Report dictionary.
    """
    passed_count = sum(
        1
        for r in results
        if r.get("validation", {}).get("overall_passed", False)
    )
    failed_count = sum(
        1
        for r in results
        if r.get("status") == "completed"
        and not r.get("validation", {}).get("overall_passed", False)
    )
    error_count = sum(1 for r in results if r.get("status") in ("error", "failed"))

    report: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_datasets": len(results),
        "passed": passed_count,
        "failed": failed_count,
        "errors": error_count,
        "results": results,
    }

    # Print summary
    print(f"\n{'='*60}")
    print("E2E TEST SUMMARY")
    print("="*60)
    print(f"Total Datasets: {report['total_datasets']}")
    print(f"Passed: {report['passed']}")
    print(f"Failed: {report['failed']}")
    print(f"Errors: {report['errors']}")
    print("="*60 + "\n")

    # Save to file
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(report, indent=2, default=str))
        print(f"Report saved to: {output_path}")

    return report


ENV_CONFIGS = {
    "test": {
        "base_url": "http://localhost:8000",
        "output_path": "backend/evals/e2e_results_test.json",
        "headers": {"X-Environment": "test"},
    },
    "dev": {
        "base_url": "http://localhost:8000",
        "output_path": "backend/evals/e2e_results_dev.json",
        "headers": {"X-Environment": "dev"},
    },
    "staging": {
        "base_url": "https://staging-api.example.com",
        "output_path": "backend/evals/e2e_results_staging.json",
        "headers": {"X-Environment": "staging"},
    },
    "production": {
        "base_url": "https://api.example.com",
        "output_path": "backend/evals/e2e_results_production.json",
        "headers": {"X-Environment": "production"},
    },
}


async def main() -> None:
    """Main entry point for E2E test CLI."""
    parser = argparse.ArgumentParser(description="E2E Test CLI for Causal Analysis API")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Run single dataset by ID (e.g., 'titanic')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 8 datasets sequentially",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="test",
        choices=list(ENV_CONFIGS.keys()),
        help="Environment to run tests against. Accepted values: test (default), dev, staging, production. "
             "Each environment has a predefined base URL, output path, and headers.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL (overrides --env setting if provided)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Analysis timeout in seconds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON report path (overrides --env setting if provided)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    if not args.dataset and not args.all:
        parser.error("Must specify --dataset or --all")

    # Get environment configuration
    env_config = ENV_CONFIGS.get(args.env, ENV_CONFIGS["test"])
    base_url = args.base_url if args.base_url else env_config["base_url"]
    output_path = args.output if args.output else env_config["output_path"]
    env_headers = env_config.get("headers", {})

    logger.info(f"Running E2E tests in '{args.env}' environment")
    logger.info(f"Base URL: {base_url}")
    logger.info(f"Output path: {output_path}")

    # Initialize client
    async with E2ETestClient(base_url=base_url) as client:
        # Setup test user
        logger.info("Setting up test user...")
        await setup_test_user(client)

        # Run tests
        if args.dataset:
            logger.info(f"Running single dataset: {args.dataset}")
            results = await run_single_dataset(client, args.dataset, args.timeout)
        else:  # args.all
            logger.info("Running all datasets...")
            results = await run_all_datasets(client, args.timeout)

        # Generate report
        report = generate_report(results, output_path)

        # Exit with appropriate code
        if report["failed"] > 0 or report["errors"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
