# Running Federated Learning MVP in Google Colab

This guide provides step-by-step instructions for running the Federated Learning MVP in Google Colab.

## 🚀 Quick Start (Recommended)

### Option 1: Use the Pre-Made Notebook

1. **Open the notebook directly in Colab:**

   Click this link: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/almosttomorrow/federatedlearning/blob/main/Federated_Learning_Colab.ipynb)

2. **Follow the notebook instructions:**
   - The notebook has all steps pre-configured
   - Simply run each cell in order
   - Wait for the public URL to appear
   - Click the URL to access the app

**That's it!** The notebook handles everything automatically.

---

## 📝 Manual Setup (Step-by-Step)

If you prefer to set it up manually or want to understand what's happening:

### Step 1: Create a New Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File > New notebook**

### Step 2: Install Dependencies

Run this in a code cell:

```python
%%capture
# Install all required packages (takes 2-3 minutes)
!pip install streamlit tensorflow numpy pandas matplotlib seaborn tenseal openai python-dotenv pyngrok
```

### Step 3: Clone the Repository

Run this in a new code cell:

```python
# Clone the repository
!git clone https://github.com/almosttomorrow/federatedlearning.git
%cd federatedlearning
```

### Step 4: Configure Environment (Optional)

Set your OpenAI API key if you want LLM explanations:

```python
import os
from getpass import getpass

# Prompt for API key (input will be hidden)
print("Enter your OpenAI API key (or press Enter to skip):")
api_key = getpass("API Key: ")

if api_key:
    # Create .env file with the API key
    with open('.env', 'w') as f:
        f.write(f'OPENAI_API_KEY={api_key}\n')
        f.write('OPENAI_MODEL=gpt-3.5-turbo\n')
    print("✓ API key configured successfully!")
else:
    print("⚠ Skipping API key configuration. LLM explanations will be disabled.")
```

**Get your API key:** https://platform.openai.com/api-keys

### Step 5: Run the Streamlit App

Run this in a new code cell:

```python
import subprocess
from pyngrok import ngrok
import time

# Kill any existing Streamlit processes
!pkill -9 streamlit

# Start Streamlit in the background
print("Starting Streamlit app...")
streamlit_process = subprocess.Popen(
    ['streamlit', 'run', 'app.py', '--server.port', '8501'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for Streamlit to start
time.sleep(5)

# Create ngrok tunnel
print("Creating public URL...\n")
public_url = ngrok.connect(8501)

print("="*70)
print("🎉 Streamlit app is running!")
print("="*70)
print(f"\n📱 Access your app at: {public_url}\n")
print("="*70)
print("\n⚠️  Keep this cell running! Stop it to shut down the app.\n")
print("💡 Tip: Click the URL above to open the app in a new tab.\n")

# Keep the cell running
try:
    streamlit_process.wait()
except KeyboardInterrupt:
    print("\n🛑 Shutting down...")
    streamlit_process.terminate()
    ngrok.kill()
```

### Step 6: Access the App

1. **Wait** for the ngrok URL to appear (looks like: `https://xxxx-xx-xx-xx-xx.ngrok-free.app`)
2. **Click** the URL to open the app in a new tab
3. **Use** the app through the Streamlit interface

---

## 🎯 Using the App

Once the app opens, follow this workflow:

### 1. Generate Synthetic Data
- Click **"🎲 Generate Synthetic Data"**
- Wait for data generation (~1-2 seconds)
- Explore the data visualizations in the expandable sections

### 2. Train Local Models
- Click **"🤖 Train Local Models"**
- Wait for training to complete (~30-60 seconds per bank)
- View the model weights visualizations

### 3. Encrypt Weights
- Click **"🔒 Encrypt Weights"**
- Wait for encryption (~5-10 seconds)
- View encryption information

### 4. Aggregate & Create Global Model
- Click **"🌐 Aggregate & Create Global Model"**
- Wait for aggregation (~5 seconds)
- View global model weights
- Download the global model (optional)

### 5. Generate Explanation (Requires API Key)
- Click **"💡 Generate Explanation"**
- Wait for LLM to generate explanation (~10-20 seconds)
- Read the comprehensive explanation

### Customization Options (Sidebar)

Adjust these parameters before generating data:
- **Number of Banks**: 2-5 (default: 3)
- **Samples per Bank**: 500-2000 (default: 1000)
- **Training Epochs**: 5-20 (default: 10)

---

## 🛑 Stopping the App

### To stop the app:

**Option 1: Interrupt the cell**
- Click **Runtime > Interrupt execution**

**Option 2: Run a stop cell**
```python
# Stop Streamlit and ngrok
!pkill -9 streamlit
from pyngrok import ngrok
ngrok.kill()
print("✓ App stopped successfully!")
```

---

## ⚠️ Important Notes

### Runtime Limits

**Google Colab Free Tier:**
- Maximum session: 12 hours
- May disconnect if idle
- GPU not required for this app (CPU is sufficient)

**If disconnected:**
- Your session will be lost
- Re-run all cells to restart the app
- You'll get a new ngrok URL

### Memory Considerations

**Colab provides:**
- ~12 GB RAM (free tier)
- ~25 GB RAM (Colab Pro)

**To avoid memory issues:**
- Use default settings first (3 banks, 1000 samples)
- Increase gradually if you need more
- Reduce settings if you get memory errors

### Security

**Your API key is safe:**
- Stored only in your Colab session
- Not shared with others
- Deleted when session ends
- Use `getpass()` to hide input

**Public URL:**
- Anyone with the ngrok URL can access your app
- URL changes each time you run
- URL expires when you stop the app

---

## 🐛 Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```python
# Re-run the installation cell
!pip install --upgrade streamlit tensorflow numpy pandas matplotlib seaborn tenseal openai python-dotenv pyngrok
```

### Issue: App won't start

**Solution:**
```python
# Kill all Streamlit processes
!pkill -9 streamlit
# Wait 10 seconds
import time
time.sleep(10)
# Re-run the "Run the Streamlit App" cell
```

### Issue: ngrok URL not working

**Solutions:**
1. Copy the URL manually and paste in a new browser tab
2. Try incognito/private browsing mode
3. Check if your company/school network blocks ngrok
4. Try a different browser

### Issue: LLM explanations not working

**Solutions:**
1. Verify your API key is correct
2. Check your OpenAI account has credits
3. Re-run Step 4 with the correct API key
4. Check the error message in the app

### Issue: Memory errors

**Solutions:**
1. Use smaller values:
   - Reduce number of banks to 2
   - Reduce samples to 500
   - Reduce epochs to 5
2. Restart the runtime:
   - **Runtime > Restart runtime**
3. Clear outputs:
   - **Edit > Clear all outputs**

### Issue: TenSEAL installation fails

**Solution:**
```python
# Install build dependencies first
!apt-get update
!apt-get install -y cmake build-essential
!pip install tenseal
```

### Issue: "Port already in use"

**Solution:**
```python
# Kill existing processes
!pkill -9 streamlit
!fuser -k 8501/tcp
# Wait and retry
```

---

## 💡 Tips & Best Practices

### Performance Tips

1. **Start small**: Use default settings first, then increase
2. **Use CPU**: GPU not needed (and may slow things down)
3. **Clear outputs**: Periodically clear cell outputs to free memory
4. **Restart when needed**: Don't hesitate to restart the runtime

### Workflow Tips

1. **Save your work**: Download the global model before stopping
2. **Take screenshots**: Capture interesting visualizations
3. **Experiment**: Try different numbers of banks and samples
4. **Read explanations**: They're educational and insightful

### Cost-Saving Tips

1. **API key**: LLM explanations cost ~$0.01-0.05 per generation
2. **Use gpt-3.5-turbo**: Much cheaper than gpt-4
3. **Skip explanations**: App works fine without them
4. **Monitor usage**: Check your OpenAI usage dashboard

---

## 🔄 Restart from Scratch

If something goes wrong, here's how to start fresh:

1. **Stop everything**:
   ```python
   !pkill -9 streamlit
   from pyngrok import ngrok
   ngrok.kill()
   ```

2. **Restart runtime**:
   - **Runtime > Restart runtime**

3. **Run all cells again**:
   - **Runtime > Run all**

---

## 📊 Expected Performance

### Timing (on Colab free tier):

| Step | Duration |
|------|----------|
| Install dependencies | 2-3 minutes |
| Clone repository | 5-10 seconds |
| Generate data | 1-2 seconds |
| Train models (3 banks) | 30-90 seconds |
| Encrypt weights | 5-10 seconds |
| Aggregate | 2-5 seconds |
| LLM explanation | 10-20 seconds |

### Resource Usage:

| Resource | Usage |
|----------|-------|
| RAM | ~2-4 GB |
| Disk | ~500 MB |
| CPU | Moderate (60-80% during training) |

---

## 🆘 Getting Help

### Still having issues?

1. **Check the main README**: https://github.com/almosttomorrow/federatedlearning/blob/main/README.md
2. **Open an issue**: https://github.com/almosttomorrow/federatedlearning/issues
3. **Check Colab docs**: https://colab.research.google.com/notebooks/
4. **Check Streamlit docs**: https://docs.streamlit.io/

---

## 🎓 Learning Resources

### Understanding the Concepts:

- **Federated Learning**: https://ai.googleblog.com/2017/04/federated-learning-collaborative.html
- **Homomorphic Encryption**: https://github.com/OpenMined/TenSEAL
- **CKKS Scheme**: https://eprint.iacr.org/2016/421.pdf

### Related Projects:

- **TensorFlow Federated**: https://www.tensorflow.org/federated
- **PySyft**: https://github.com/OpenMined/PySyft
- **Flower**: https://flower.dev/

---

## ✅ Success Checklist

Before considering your setup complete, verify:

- [ ] All dependencies installed without errors
- [ ] Repository cloned successfully
- [ ] Streamlit app starts and shows URL
- [ ] Can access the app through ngrok URL
- [ ] Can generate synthetic data
- [ ] Can train local models
- [ ] Can encrypt weights
- [ ] Can create global model
- [ ] (Optional) LLM explanations work with API key

---

**Happy Learning!** 🚀

If you found this helpful, consider starring the repository on GitHub!
