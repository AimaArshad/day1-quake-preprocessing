# Day 1 Lab: Answers

Name:

## 1. Leakage

Which columns did you drop and why? Is `tsunami` leaky? Is `magType`?

Dropped: title, sig, mmi, cdi, felt, alert, tsunami, mag, magType, magError, magNst.
Reason: none of them are known at prediction time, and most are direct functions of the magnitude (the target). Test applied: "Would I know this when I predict?"
tsunami — leaky. The flag is set by staff after the event, once a big quake has already occurred. Seeing it = seeing a consequence of the label.
magType — leaky. The method (ml, mb, mww…) correlates with magnitude band (mww → big, md → small) and is only assigned once the magnitude is computed.

## 2. Stream vs batch

How many events changed (same `id`, newer `updated`) during your stream window? What does that tell you about "latest version wins"?

Over a 40-min window on all_hour: a few events got revised (same id, newer updated), most polls had 0–2 new pairs.
Takeaway: id alone isn't a stable identity — the same event returns with corrections. Using (id, updated) as the key collects every version; dedupe_latest then picks the newest. Stream collects versions, batch picks the winner.

## 3. Outliers

Your decision on negative depth and negative magnitude, with reasoning.
Negative depth — keep. Real measurements above the reference datum (volcanic/geothermal/mountainous shallow quakes). Physically plausible, not an error.
Negative magnitude — keep. Magnitude is a log scale; "negative" just means very small. Real detections on sensitive local networks.
Rule: IQR flags unusual, not wrong. Delete only if unusual and physically implausible. Neither is.

## 4. Cardinality

You grouped `region` to top-k. Name one alternative encoding and one risk it carries.

Alternative: target (mean) encoding — replace each region with its mean big_quake rate.
Risk: target leakage. The encoding is computed from the label, so pre-split computation lets the model memorise the target. Fix: compute inside CV folds (out-of-fold) and smooth toward the global mean.
Top-k + one-hot is weaker but unsupervised and leakage-safe — the right default here.
