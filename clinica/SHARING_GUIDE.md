# 📤 Guide: Sharing Your Clinical Analysis Notebook with Reviewers

## ✅ Best Options (Ranked)

### **Option 1: HTML Export (RECOMMENDED) ✨**
**Status:** ✅ Already created: `explanation.html`

**Advantages:**
- ✅ All images embedded as base64 (truly self-contained, ~15 MB file with 135 images)
- ✅ Opens in any web browser (Chrome, Firefox, Safari, Edge)
- ✅ Clean, professional appearance (code cells hidden)
- ✅ Maintains formatting, equations, and styling
- ✅ Works when downloaded/moved to any location
- ✅ Easy to email or share via cloud storage

**How to share:**
```bash
# Send the HTML file via:
# - Email attachment (if under 25MB limit)
# - Google Drive / Dropbox link (recommended for large files)
# - GitHub repository
# - University file sharing system (e.g., WeTransfer)
```

**Regenerate if needed:**
```bash
cd /home/zano/Documents/TESI/FOLDER_CINECA
source .venv/bin/activate
# IMPORTANT: Use --embed-images flag to embed all images
python -m jupyter nbconvert --to html --no-input --embed-images clinica/explanation.ipynb
```

---

### **Option 2: PDF Export (Professional)**
**Status:** ⚠️ Requires pandoc installation

**Advantages:**
- ✅ Professional document format
- ✅ Print-ready for thesis committees
- ✅ Preserves pagination and layout

**Install pandoc (if needed):**
```bash
sudo apt install pandoc texlive-xetex texlive-fonts-recommended texlive-plain-generic
```

**Then export:**
```bash
cd /home/zano/Documents/TESI/FOLDER_CINECA
source .venv/bin/activate
python -m jupyter nbconvert --to pdf --no-input clinica/explanation.ipynb
```

---

### **Option 3: Share Entire Repository**
**Best for:** Collaborators who want to run the code themselves

**Via Git:**
```bash
# If already on GitHub/GitLab
git add .
git commit -m "Add clinical data exploration notebook"
git push
# Share repository URL
```

**Via ZIP archive:**
```bash
cd /home/zano/Documents/TESI/FOLDER_CINECA
zip -r clinical_analysis.zip clinica/ images/ style/
# Send clinical_analysis.zip
```

---

### **Option 4: Embed Images Directly (Advanced)**
Convert external images to base64-encoded inline images (makes notebook self-contained).

**Not recommended** because:
- ❌ Makes `.ipynb` file very large (several MB)
- ❌ Harder to maintain/update images
- ❌ Version control becomes messy

---

## 🎯 Quick Decision Matrix

| Reviewer needs...              | Best option                |
|-------------------------------|----------------------------|
| Just read the results         | ✅ HTML (explanation.html) |
| Print for thesis review       | PDF (requires pandoc)      |
| Run code / modify analysis    | Full repo (ZIP or Git)     |
| Formal thesis submission      | PDF or LaTeX integration   |

---

## 📧 Email Template for Reviewers

```
Subject: Clinical Data Analysis - MSA vs PD Classification

Dear [Reviewer Name],

Please find attached the clinical data exploration report for my master's 
thesis on differentiating Parkinson's Disease from Multiple System Atrophy.

File: explanation.html (320 KB, self-contained)
- Open with any web browser
- All figures and analysis results embedded
- Clean, thesis-quality formatting

Key findings:
- Symptom prevalence and co-occurrence patterns
- L-dopa responsiveness differential (key diagnostic criterion)
- Feature importance analysis (10 discriminative clinical variables)
- ML baseline models (Random Forest achieves X% accuracy on 8-fold CV)

I look forward to your feedback.

Best regards,
Luca Zanotto
```

---

## 🔄 Keeping Files in Sync

If you update the notebook or regenerate figures:

```bash
# 1. Update figures in ../images/
# 2. Regenerate HTML
cd /home/zano/Documents/TESI/FOLDER_CINECA
source .venv/bin/activate
python -m jupyter nbconvert --to html --no-input clinica/explanation.ipynb

# 3. Share the new HTML file
```

---

## 📊 Current File Sizes

- `explanation.ipynb`: ~15 KB (references external images via relative paths)
- `explanation.html`: **~15 MB** (135 images embedded as base64, truly self-contained) ✅
- `../images/`: ~50+ PNG figures (separate files)

**Recommendation:** Always share the **HTML version** (`explanation.html`) for reviews. 
The file will display correctly even when downloaded to a different location, as all 
images are embedded inside the HTML.

