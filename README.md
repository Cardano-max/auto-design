# WhatsApp-ChatGPT Marketing Image Generator

This project automates the creation of professional marketing images by connecting WhatsApp input with ChatGPT's image generation capabilities. It's designed to transform basic product photos into marketing-ready materials with branding elements like company name, product name, price, tagline, and more.

## Features

- **Web-based Demo Interface** that simulates WhatsApp conversations
- **ChatGPT Automation** using Selenium WebDriver
- **Prompt Engineering** templates optimized for different product types
- **Image Processing** for uploading, downloading, and managing images
- **Session Management** to track user conversations and preferences

## Requirements

- Python 3.8+
- Chrome Browser
- ChromeDriver (automatically installed by webdriver-manager)
- Flask
- Selenium

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/whatsapp-chatgpt-marketing.git
   cd whatsapp-chatgpt-marketing
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install selenium webdriver-manager flask pillow
   ```

4. Create directories for templates and static files:
   ```
   mkdir -p templates static/images
   ```

5. Copy the `templates/index.html` file to the templates directory.

## Usage

1. Start the application:
   ```
   python auto_marketing.py
   ```

2. Open your browser and navigate to `http://localhost:5000`

3. Click "Initialize Bot" to start the ChatGPT automation (this will open a Chrome window)

4. If prompted, log in to your ChatGPT account

5. Upload a product image and provide details (company name, product name, price, etc.)

6. Click "Generate Marketing Image" to create your marketing material

## How It Works

1. The user uploads a product image and provides details through the web interface
2. The Python script automates browser interaction with ChatGPT:
   - Opens a browser session to ChatGPT
   - Uploads the product image
   - Sends a carefully crafted prompt based on the product type and details
   - Waits for ChatGPT to generate the marketing image
   - Downloads the generated image
3. The generated image is displayed in the chat interface
4. Images are saved locally for future reference

## Project Structure

- `auto_marketing.py` - Main Python script handling automation and web server
- `templates/index.html` - Web interface simulating WhatsApp
- `images/input/` - Directory for storing uploaded product images
- `images/output/` - Directory for storing generated marketing images

## Customization

### Prompt Templates

You can modify the prompt templates in the `PromptTemplates` class in `auto_marketing.py` to change how the marketing images are generated. There are templates for:
- General products
- Beverages
- Food items

### CSS Selectors

If ChatGPT's interface changes, you may need to update the CSS selectors in the `CHATGPT_SELECTORS` dictionary.

## Production Deployment Considerations

For production use:
1. Set headless mode to `True` in the browser initialization
2. Implement proper error handling and retries
3. Add user authentication to the web interface
4. Set up proper logging and monitoring
5. Use a production-ready web server instead of Flask's development server
6. For real WhatsApp integration, implement the Meta Cloud API connection

## Troubleshooting

- **ChatGPT Login Issues**: You may need to manually log in the first time. The script will wait for you to complete authentication.
- **Selector Errors**: If ChatGPT's interface changes, update the CSS selectors in the `CHATGPT_SELECTORS` dictionary.
- **Image Generation Failures**: Sometimes ChatGPT may not generate an image on the first try. The script will attempt to explicitly request an image if none is found initially.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Extending the Project

To implement actual WhatsApp integration with Meta Cloud API:
1. Register for WhatsApp Business API access
2. Set up webhook endpoints for receiving messages
3. Modify the code to handle incoming webhook events
4. Update the image sending function to use the WhatsApp API