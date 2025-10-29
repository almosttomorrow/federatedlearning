# 🚀 Quick Start: Google Colab (5 Minutes)

The fastest way to run the Federated Learning MVP!

---

## Method 1: One-Click Launch (Recommended)

### Step 1: Click the Button

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/almosttomorrow/federatedlearning/blob/main/Federated_Learning_Colab.ipynb)

### Step 2: Run All Cells

In Colab, click: **Runtime > Run all**

### Step 3: Set API Key (Optional)

When prompted, enter your OpenAI API key or press Enter to skip.

### Step 4: Access the App

Click the ngrok URL that appears (looks like `https://xxxx.ngrok-free.app`)

### Step 5: Use the App

Follow the 5-step workflow in the UI!

**That's it!** ✅

---

## Method 2: Manual Setup (3 Steps)

### Step 1: Create New Notebook

Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

### Step 2: Copy & Paste This Code

```python
# Install and setup (2-3 minutes)
!pip install -q streamlit tensorflow numpy pandas matplotlib seaborn tenseal openai python-dotenv pyngrok

# Clone repo
!git clone https://github.com/almosttomorrow/federatedlearning.git
%cd federatedlearning

# Set API key (optional)
import os
from getpass import getpass
api_key = getpass("OpenAI API Key (or press Enter): ")
if api_key:
    with open('.env', 'w') as f:
        f.write(f'OPENAI_API_KEY={api_key}\n')

# Run app
import subprocess
from pyngrok import ngrok
import time

!pkill -9 streamlit
process = subprocess.Popen(['streamlit', 'run', 'app.py', '--server.port', '8501'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(5)
url = ngrok.connect(8501)
print(f"\n\n🎉 App URL: {url}\n\n")
process.wait()
```

### Step 3: Run the Cell

Click the play button or press `Shift + Enter`

---

## 📱 What You'll See

1. **Installation progress** (2-3 minutes)
2. **Repo cloned** ✓
3. **API key prompt** (optional)
4. **Public URL** - Click this!
5. **Streamlit app** opens in new tab

---

## 🎯 Using the App

Once the app opens:

1. Click **"Generate Synthetic Data"**
2. Click **"Train Local Models"**
3. Click **"Encrypt Weights"**
4. Click **"Aggregate & Create Global Model"**
5. Click **"Generate Explanation"** (requires API key)

Use the **sidebar** to adjust settings (number of banks, samples, epochs).

---

## ⚠️ Important

- **Keep the Colab cell running** - Stopping it will shut down the app
- **Free tier limits** - 12 hours max session
- **URL expires** - When you stop the app
- **API costs** - LLM explanations cost ~$0.01-0.05 each

---

## 🐛 Quick Fixes

**App won't start?**
```python
!pkill -9 streamlit
# Wait 10 seconds, then re-run
```

**URL not working?**
- Copy and paste manually
- Try incognito mode

**Memory error?**
- Use 2 banks instead of 3
- Use 500 samples instead of 1000

---

## 📚 Need More Help?

- **Detailed Guide**: [COLAB_SETUP.md](COLAB_SETUP.md)
- **Full README**: [README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/almosttomorrow/federatedlearning/issues)

---

**Total Time: 5 minutes** ⏱️

**Difficulty: Easy** ✅

**Cost: Free** 💰 (except optional LLM explanations)

---

Enjoy exploring federated learning! 🚀
