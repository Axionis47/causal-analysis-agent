# User Guide

Welcome to the Causal Analysis Agent. This guide walks you through using the system to analyze datasets and discover causal relationships.

---

## What You Need

Before starting, make sure you have:

- A Kaggle account (free at kaggle.com)
- A dataset URL from Kaggle
- Access to the application (ask your admin for credentials)

---

## Step 1: Create Your Account

Open the application in your browser. Click "Sign Up" and enter your email and password. You'll get a confirmation message. Now click "Log In" and enter your credentials. You're in.

![Sign up screen](assets/screenshot-signup.png)
*The sign up form where you enter your email and password*

---

## Step 2: Find a Dataset

Go to kaggle.com and find a dataset you want to analyze. Look for datasets with:

- At least 100 rows
- Multiple numeric columns
- Not too many missing values

Copy the dataset URL. It should look like this:

```
https://www.kaggle.com/datasets/username/dataset-name
```

![Dataset selection on Kaggle](assets/screenshot-dataset-selection.png)
*Finding and copying a dataset URL from Kaggle*

Some good datasets for trying out causal analysis:

- Iris dataset: `https://www.kaggle.com/datasets/uciml/iris`
- Adult income: `https://www.kaggle.com/datasets/uciml/adult-census-income`
- Boston housing: `https://www.kaggle.com/datasets/vikrishnan/boston-house-prices`

---

## Step 3: Preview Your Data

Before running a full analysis, check if your dataset is suitable. Click "Preview Data Quality" and paste your Kaggle URL. The system downloads a sample and shows you:

- How many rows and columns
- Any quality issues (missing values, duplicates, etc.)
- Estimated analysis time

![Data preview screen](assets/screenshot-preview.png)
*The data quality preview showing dataset statistics and potential issues*

If you see warnings, read them carefully. Some warnings can be ignored by checking "Override warnings" when creating the analysis. For example:

- 10% missing values? Usually okay to proceed.
- 50% missing values? Might want to pick a different dataset.
- Only 50 rows? Too small, pick something with more data.

---

## Step 4: Create an Analysis

Click "New Analysis" and paste your Kaggle URL. The system asks what kind of analysis you want:

![New analysis form](assets/screenshot-analysis-run.png)
*The analysis creation form with analysis type options*

**Analysis Types:**

- **Treatment Effects**: Find out if X causes Y. For example, "Does education affect income?"
- **Causal Discovery**: Let the system find all causal relationships automatically.
- **Mediation**: See if X affects Y through Z. For example, "Does education affect income through job type?"
- **Heterogeneous Effects**: Check if effects differ across groups. For example, "Is the effect of education on income different for men and women?"

If you're not sure, leave it on "Auto" and the system picks the best methods for your data.

---

## Step 5: Watch Progress

After clicking "Start Analysis", you'll see a progress bar. The system goes through several stages:

1. **Downloading** your dataset from Kaggle
2. **Analyzing** data characteristics (column types, distributions, quality)
3. **Finding** causal relationships (the interesting part)
4. **Estimating** effect sizes (how strong are the relationships)
5. **Validating** results (making sure findings are reliable)
6. **Generating** reports (putting it all together)

This takes a few minutes depending on dataset size. You can close the browser and come back later. Your analysis keeps running.

---

## Step 6: View Results

When the analysis finishes, you'll see four tabs:

![Results view](assets/screenshot-results-view.png)
*The results dashboard showing all four tabs: Summary, Technical, Visualizations, and Validation*

### Summary Tab

This is the executive summary. It explains what the system found in plain English. Look for:

- **Key findings**: What causes what
- **Effect sizes**: How strong the relationships are
- **Confidence levels**: How sure we are

Example finding: "Education has a significant positive effect on income. Each additional year of education increases income by approximately $5,000."

### Technical Tab

This shows the methods used and detailed statistics. Useful if you want to understand how the analysis works or need to cite specific methods.

You'll see things like:
- Which algorithms were used (PC, GES, FCI)
- Statistical test results (p-values, confidence intervals)
- Data preprocessing steps applied

### Visualizations Tab

This shows graphs and charts:

- **Causal graph**: Arrows show which variables affect others. Follow the arrows to trace cause and effect.
- **Effect plots**: Show treatment effects with confidence intervals. The wider the interval, the less certain we are.
- **Distribution plots**: Show your data characteristics.

You can click on the causal graph to edit it. Add or remove arrows if you know something the system missed. Maybe you know from your domain that X definitely causes Y, but the algorithm didn't find it. Add that arrow manually.

### Validation Tab

This shows robustness checks. The system runs tests to make sure the findings are reliable:

- **Green checks**: Results passed validation. Good news.
- **Yellow warnings**: Be cautious. The finding might not be as solid as it seems.
- **Red errors**: The results might not be trustworthy. Read the details to understand why.

---

## Step 7: Download Reports

Click "Download" and pick a format:

![Download options](assets/screenshot-download.png)
*The download menu showing available report formats*

- **PDF**: Nice looking report for presentations
- **Markdown**: Text format for documentation
- **HTML**: Web page you can share
- **PowerPoint**: Slides for meetings
- **JSON**: Raw data for further analysis

The PDF is great for sharing with colleagues who want the highlights. The JSON is useful if you want to do your own analysis with the raw results.

---

## Step 8: Share Your Analysis

Want to show your results to someone? Click "Share" and you'll get a link. You can set:

![Share dialog](assets/screenshot-share.png)
*The share dialog where you configure link expiration and access type*

- **Expiration**: How long the link works (1-365 days)
- **Access type**: Public (anyone can view) or Private (requires login)

The system tracks how many times your link is viewed. You can revoke the link anytime if you change your mind.

---

## Advanced Features

### Testing Hypotheses

Found an interesting relationship? Test what happens if you change variables. Click "Test Hypothesis" and pick different treatment, outcome, or confounders.

For example, if your original analysis looked at education -> income, you could test:
- Does work experience -> income?
- Does education -> income controlling for age?

The system creates a new analysis with your settings. Compare the results to see if your hypothesis holds up.

### Comparing Analyses

Run multiple analyses and compare them. Click "Compare" and select 2-3 analyses. The system shows:

- How similar the causal graphs are
- Differences in effect estimates
- Which analysis is more reliable

This is useful for sensitivity analysis. If two analyses with different settings give similar results, you can be more confident in the findings.

### Version History

Every time you change analysis settings, the system saves a version. Click "History" to see all versions. You can:

- Compare any two versions
- See what changed
- Revert to an old version

This is like "undo" but better. You can go back to any previous state.

### Adding Comments

Collaborate with your team by adding comments. Click anywhere in the results and type your thoughts. Others can reply to your comments. Use @mentions to notify specific people.

Comments are great for:
- Noting observations
- Asking questions
- Recording decisions
- Tracking follow-up items

Mark comments as "resolved" when you've addressed them.

### Preprocessing Options

By default, the system cleans your data automatically:

- Fills in missing values (using median for numbers, mode for categories)
- Removes outliers (values more than 3 standard deviations from the mean)
- Encodes categories as numbers (one-hot encoding)
- Scales numeric columns (standardization)

If you want raw data instead, uncheck "Enable preprocessing" when creating the analysis. Or click "Preview Preprocessing" to see what changes before running.

---

## Tips for Better Results

1. **Use datasets with clear cause and effect relationships.** Survey data about customer satisfaction might work better than random web scraping.

2. **Include potential confounders.** Variables that affect both treatment and outcome should be in your dataset. For example, if studying education -> income, include age (which affects both).

3. **Avoid datasets with too many missing values.** More than 50% missing is problematic. The system can handle some missing data, but too much makes results unreliable.

4. **Check the validation tab.** Green checks mean you can trust the results. Yellow or red means be careful about drawing conclusions.

5. **Run multiple analyses with different settings.** If results are consistent across different methods, you can be more confident.

6. **Read the technical report.** Understanding the methodology helps you interpret results correctly and explain them to others.

---

## Common Questions

**How long does analysis take?**

Usually 2-10 minutes. Large datasets (over 100,000 rows) take longer. Very large datasets get sampled automatically to keep things reasonable.

**Can I run multiple analyses at once?**

Yes, up to 3 at a time. After that, you'll need to wait for one to finish.

**What if my analysis fails?**

Check the error message. Common issues are invalid Kaggle URLs, unsupported file types, or data quality problems. Try the preview endpoint first to catch issues early.

**How much does it cost?**

The system uses AI models that cost money. Admins set quotas to control costs. You can see your usage in your profile. Contact your admin if you need higher limits.

**Can I use my own data?**

Currently only Kaggle datasets are supported. Upload your data to Kaggle first (you can make it private), then use the URL.

**What if I disagree with the causal graph?**

You can edit it manually. Click on the graph and add or remove arrows. The system saves your changes as a new version, so you can always go back.

**How do I know if results are trustworthy?**

Check the validation tab. Look for green checks. Read the confidence scores. Higher is better. Compare with what you know about the domain. If the results match your intuition and pass validation, they're probably good.

**Can I export data for my own analysis?**

Yes, download the JSON format. It includes all raw results, statistics, and metadata. You can load this into Python, R, or any other tool.

---

## Need Help?

If you're stuck:

1. Check the troubleshooting guide: `docs/TROUBLESHOOTING.md`
2. Review API documentation: `/api/docs`
3. Look at the architecture overview: `docs/ARCHITECTURE.md`
4. Contact your admin for access issues
5. Report bugs at: https://github.com/anthropics/claude-code/issues

Happy analyzing!
