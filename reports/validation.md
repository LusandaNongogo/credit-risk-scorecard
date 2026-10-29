# Model Validation Report: Application PD Scorecard

## 1. Purpose And Scope

This report validates a logistic regression probability of default (PD) scorecard built on the German Credit data. It covers the data, the method, discrimination, stability, calibration and limitations, and closes with what would need to change to use a model like this in South Africa.

The model is a portfolio piece, not a production model. The aim is to show the full build and validation process end to end.

## 2. Data

| Item | Value |
| --- | --- |
| Source | German Credit data, loaded with `scorecardpy.germancredit()` |
| Rows | 1 000 loans |
| Target | `creditability`, bad = 1 |
| Bad rate | 30.0% |
| Features available | 20 |
| Features excluded up front | `personal_status_and_sex`, `foreign_worker` |
| Split | Stratified 70/30, seed 42 (700 train, 300 test, both 30% bad) |

**Exclusions.** `personal_status_and_sex` combines marital status with sex, and `foreign_worker` stands in for nationality. Both are protected characteristics. Using them in a credit decision would be discriminatory, and in South Africa it would conflict with the Constitution's equality clause and the Promotion of Equality and Prevention of Unfair Discrimination Act. They were removed before binning so they could not influence feature selection.

**Data quality.** No missing values. Amounts are in Deutsche Mark. The data dates from the 1970s to 1990s and the 30% bad rate is far above a normal loan book, which suggests bads were oversampled.

## 3. Method

1. **Binning.** Fine and coarse classing with `scorecardpy.woebin` on train only. WoE is ln(%bad / %good), so a positive WoE marks a riskier bin. The WoE and IV calculation was checked against an independent implementation in `src/woe.py` and matches to six decimal places.
2. **Feature selection by IV.** Features with IV below 0.02 were dropped. Anything above 0.5 was flagged for review.
3. **Model.** Logistic regression (statsmodels) on the WoE values of the kept features.
4. **Sign check.** Every coefficient must be positive under this WoE convention. The fitting routine would drop the worst wrong signed feature and refit, but none were needed.
5. **Scaling.** 600 points at good:bad odds of 50:1, 20 points to double the odds. Factor = 20 / ln 2 = 28.85, offset = 487.12.

### Information Value

| Feature | IV | Decision |
| --- | ---: | --- |
| status_of_existing_checking_account | 0.624 | Keep, flagged above 0.5 |
| duration_in_month | 0.259 | Keep |
| credit_history | 0.254 | Keep |
| savings_account_and_bonds | 0.232 | Keep |
| credit_amount | 0.201 | Keep |
| age_in_years | 0.142 | Keep |
| property | 0.140 | Keep |
| purpose | 0.130 | Keep |
| present_employment_since | 0.130 | Keep |
| other_installment_plans | 0.117 | Keep |
| housing | 0.102 | Keep |
| installment_rate_in_percentage_of_disposable_income | 0.034 | Keep |
| job, telephone, number_of_existing_credits_at_this_bank, present_residence_since, other_debtors_or_guarantors, number_of_people_being_liable_to_provide_maintenance_for | below 0.007 | Drop |

**Flagged feature.** Checking account status has an IV of 0.62. The bad rate falls steadily across its bins (47.9% overdrawn, 37.1% low or positive balance, 11.6% no checking account), which is a sensible pattern and not a sign of target leakage. It is kept. In production I would confirm the field is captured at application and never updated after the outcome is known.

### Coefficients

| Feature (WoE) | Coefficient | p value |
| --- | ---: | ---: |
| const | -0.867 | 0.000 |
| status_of_existing_checking_account | 0.785 | 0.000 |
| duration_in_month | 0.743 | 0.001 |
| credit_history | 0.621 | 0.002 |
| savings_account_and_bonds | 0.820 | 0.000 |
| credit_amount | 1.032 | 0.000 |
| age_in_years | 0.761 | 0.006 |
| property | 0.269 | 0.382 |
| purpose | 1.043 | 0.000 |
| present_employment_since | 0.685 | 0.013 |
| other_installment_plans | 0.850 | 0.004 |
| housing | 0.609 | 0.074 |
| installment_rate_in_percentage_of_disposable_income | 1.749 | 0.002 |

All signs are positive. `property` and `housing` are not significant at 5%, probably because they overlap (owning property and owning a home carry similar information). They were kept for this version because their IV is reasonable and their signs are right; a leaner card would drop them first.

## 4. Discrimination

| Sample | AUC | Gini | KS |
| --- | ---: | ---: | ---: |
| Train | 0.835 | 0.669 | 0.537 |
| Test | 0.799 | 0.597 | 0.511 |

A test Gini near 0.60 is within the normal range for an application scorecard. The drop from train to test (0.07 Gini) is modest given only 300 test rows. KS on test peaks at a score of about 516.

Charts: `figures/roc.png`, `figures/ks.png`, `figures/score_distribution.png`.

## 5. Stability

PSI between train scores (baseline, decile bins) and test scores is **0.062**, below the 0.1 threshold for a stable population. The largest single bin contribution is 0.023.

This is expected because both samples come from the same data. The meaningful use of PSI is ongoing monitoring of new applicants against this baseline, together with characteristic level stability for each feature.

## 6. Calibration

Predicted vs actual bad rate by test score decile (decile 1 = lowest scores):

| Decile | Score range | Predicted | Actual |
| ---: | --- | ---: | ---: |
| 1 | 401 to 465 | 0.787 | 0.733 |
| 2 | 467 to 480 | 0.604 | 0.467 |
| 3 | 481 to 499 | 0.464 | 0.433 |
| 4 | 501 to 512 | 0.346 | 0.600 |
| 5 | 514 to 525 | 0.245 | 0.233 |
| 6 | 526 to 534 | 0.187 | 0.267 |
| 7 | 535 to 546 | 0.142 | 0.100 |
| 8 | 547 to 563 | 0.094 | 0.067 |
| 9 | 563 to 581 | 0.053 | 0.067 |
| 10 | 582 to 634 | 0.022 | 0.033 |

Overall level is good: mean predicted PD 0.294 against an actual bad rate of 0.300. Rank ordering holds except at decile 4, which is about three standard errors above its prediction (18 bads from 30). With ten deciles of 30 loans each, one result like this can happen by chance and its neighbours are well calibrated, but it should be re-tested on a larger sample.

**Important:** the PDs are calibrated to a 30% bad rate. Before using them for pricing, cutoffs or provisions they need to be recalibrated to the real portfolio default rate, for example by adjusting the intercept.

## 7. Limitations

- **Small sample.** 700 training rows limits how finely the bins can be cut and makes test metrics noisy. Confidence intervals on Gini would be wide.
- **Old and foreign data.** The loans are decades old and German. Borrower behaviour, products and the economy have all changed, and nothing here says the relationships hold in South Africa today.
- **Oversampled bads.** The 30% bad rate distorts the PD level (see section 6).
- **No time dimension.** There is no application date, so there is no out of time test, which is the validation that matters most in practice.
- **No reject inference.** Only accepted loans have outcomes. The model has never seen the applicants who were turned down.
- **Unclear target definition.** The data does not say what counts as bad or over what window, which matters for IFRS 9 and Basel use.
- **Age is used.** Age is a legitimate and common credit variable, but it is also a protected ground under South African equality law. Its use would need a documented business justification and a check that it does not produce unfair outcomes.

## 8. Using This In South Africa

### Regulation

- **National Credit Act (NCA).** Lenders must assess affordability before granting credit. A loan granted without a proper assessment can be declared reckless and the court can suspend or set aside the debt. A PD score on its own does not satisfy this. `src/affordability.py` adds a check based on the NCR Affordability Assessment Regulations: use the higher of declared expenses and the regulated minimum expense norm for the applicant's income band, subtract existing debt repayments, and only approve if the new instalment fits. The norm values should be checked against the current NCR schedule before any real use.
- **POPIA.** Personal information must be processed lawfully, for a specific purpose and kept to what is necessary. Special personal information, which includes race, religion, health and sex life, cannot be used without a specific legal basis. Applicants also have rights around automated decisions, so the lender must be able to explain a decline, which a points based scorecard makes straightforward.
- **Equality.** As above, protected characteristics and close proxies for them stay out of the model.

### IFRS 9

South African banks provision under IFRS 9, so a PD model feeds directly into expected credit loss: ECL = PD x LGD x EAD. `src/ecl.py` implements a simplified version with three stages (12 month ECL for stage 1, lifetime ECL for stage 2, PD of 1 for stage 3) using a constant hazard for lifetime PD and yearly discounting.

On the test book, using an illustrative LGD of 45% and treating the model PD as a 12 month PD, 48 of 300 loans move to stage 2 on a PD trigger. Their coverage is 37% against 10% for stage 1, which shows how much the staging rule drives provisions. These numbers are inflated by the 30% bad rate and are there to show the mechanics only. A real IFRS 9 model also needs forward looking macroeconomic scenarios, amortising EAD and a proper PD term structure.

### Rebuilding On Local Data

1. **Bureau data.** Rebuild on South African bureau data (TransUnion, Experian, XDS) linked to the lender's own outcomes, with a clear bad definition such as 90 days past due within 12 months.
2. **Out of time validation.** Hold back the most recent cohorts as the test set.
3. **Recalibration.** Calibrate to the actual default rate of the target portfolio and recheck by segment.
4. **Thin file customers.** A large share of South African adults have little or no formal credit history. Alternative data can fill the gap: mobile airtime and data spend, mobile money and bank transaction patterns, utility and rent payments, and for informal or creative workers, records of income that is already owed to them. Any such source needs consent under POPIA and a fairness review.
5. **Monitoring.** Monthly PSI and characteristic stability, quarterly Gini and calibration, and a documented trigger for redevelopment.
