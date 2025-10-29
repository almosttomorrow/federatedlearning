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
!pip install -q streamlit tensorflow numpy pandas matplotlib seaborn tenseal openai python-dotenv
!npm install -g localtunnel

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

# Run app with localtunnel
import subprocess
import time
import threading

!pkill -9 streamlit
process = subprocess.Popen(['streamlit', 'run', 'app.py', '--server.port', '8501', '--server.headless', 'true'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(10)

lt_process = subprocess.Popen(['lt', '--port', '8501'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def read_url():
    for line in iter(lt_process.stdout.readline, ''):
        if 'your url is:' in line.lower():
            url = line.split('is:')[-1].strip()
            print(f"\n\n🎉 App URL: {url}\n\n")
            print("Note: You may need to click 'Click to Continue' on the warning page\n")
            break

url_thread = threading.Thread(target=read_url)
url_thread.start()
url_thread.join(timeout=30)
process.wait()
```

### Step 3: Run the Cell

Click the play button or press `Shift + Enter`

**Alternative: If localtunnel doesn't work, use ngrok:**
1. Sign up for free at https://dashboard.ngrok.com/signup
2. Get your authtoken at https://dashboard.ngrok.com/get-started/your-authtoken
3. See the notebook Step 5 for ngrok setup code

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
!pkill -9 node
# Wait 10 seconds, then re-run
```

**URL not working?**
- Click "Click to Continue" on the warning page
- Copy and paste URL manually
- Try incognito mode
- Use ngrok method instead (see notebook Step 5)

**ngrok authentication error?**
- Sign up at https://dashboard.ngrok.com/signup
- Get your free authtoken
- Use Step 5 in the notebook

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
