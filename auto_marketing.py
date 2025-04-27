#!/usr/bin/env python3
"""
Marketing Image Generator - API Version
-------------------------------------
Production version with WhatsApp Cloud API integration
Configured for Railway.com deployment
"""

import os
import base64
import json
import logging
import argparse
import urllib.request
import requests
from datetime import datetime
from typing import Optional, Dict, List
from openai import OpenAI
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("marketing_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MarketingBotAPI")

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Create directories for storing images
os.makedirs('images/input', exist_ok=True)
os.makedirs('images/output', exist_ok=True)

# Store user sessions
user_sessions = {}

# Initialize OpenAI client with your API key
OPENAI_API_KEY = "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"
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# WhatsApp Meta Cloud API configuration
WHATSAPP_API_VERSION = os.getenv('WHATSAPP_API_VERSION', 'v17.0')
WHATSAPP_API_URL = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}"
WHATSAPP_ACCESS_TOKEN = os.getenv('ACCESS_TOKEN', 'EAAdtq4qJH50BOZBvWMs9gF92PzZARYKPPPO3xiyy5Qj5TgLrm5ZA5xD10x28wZBexPMrdrZBiZCUTAjCU9x07hV1wpFFypjdAU30IiccM7ZBxa7ZAKmqhNFZB3oNfCK3SaIdNNvbaE2JKDuZCAeuuaBVKiFXeSyNEBGOJBqUHhpwFrpKmZBwBgylW7x6tQNfS2ZBMgZDZD')
WHATSAPP_PHONE_NUMBER_ID = os.getenv('PHONE_NUMBER_ID', '607253892466255')
WHATSAPP_VERIFY_TOKEN = os.getenv('VERIFY_TOKEN', '1122')

###################
# PROMPT TEMPLATES
###################

class PromptTemplates:
    @staticmethod
    def get_master_template(product_details: Dict) -> str:
        """Generate the master prompt template with product details for GPT Image 1"""
        return f"""Create a professional marketing poster for a product with these specifications:

1. PRODUCT PRESENTATION:
   - Place the product as the central focal point
   - Remove the original background completely
   - Apply professional product shadows for dimension

2. BRANDING ELEMENTS:
   - Company name: "{product_details.get('company_name', '')}" positioned at the top in an elegant, readable font
   - Product name: "{product_details.get('product_name', '')}" below the product image in bold typography
   - Price: "{product_details.get('price', '')}" displayed in a circular badge, bottom right corner using maroon/red with white text
   - Include a tagline: "{product_details.get('tagline', '')}" in italic font between company name and product image
   - Location/address: "{product_details.get('address', '')}" in smaller font below the product name

3. DESIGN STYLE:
   - Professional marketing aesthetics similar to high-end brands
   - Clean, modern layout with balanced spacing
   - Background: {product_details.get('background_style', 'light beige')} with subtle texture/gradient
   - Use complementary colors that enhance the product's appeal
   - Ensure all text is perfectly readable

The final image should look like it was created by an expert graphic designer for a premium brand's marketing campaign.
"""

    @staticmethod
    def get_beverage_template(product_details: Dict) -> str:
        """Generate beverage-specific prompt template for GPT Image 1"""
        return f"""Create a premium café marketing poster for a beverage with these specifications:

1. PRODUCT ENHANCEMENT:
   - Center the beverage as the focal point
   - Remove the original background completely
   - Keep whipped cream, garnishes, and toppings visible and appetizing
   - Add subtle steam effects for hot drinks or condensation for cold drinks
   - Enhance beverage colors for visual appeal

2. BRANDING PLACEMENT:
   - Company: "{product_details.get('company_name', '')}" at top in elegant script font
   - Product: "{product_details.get('product_name', '')}" below the product
   - Price: "{product_details.get('price', '')}" in a circular price tag, bottom corner
   - Tagline: "{product_details.get('tagline', '')}" below company name
   - Location: "{product_details.get('address', '')}" in smaller font at bottom

3. AESTHETIC ELEMENTS:
   - Warm background color that complements the drink
   - Subtle coffee beans/relevant ingredients in background
   - Soft vignette effect to focus on the drink
   - Style similar to Starbucks or premium coffee shop marketing

Create a professional marketing poster as described above.
"""

    @staticmethod
    def get_food_template(product_details: Dict) -> str:
        """Generate food-specific prompt template for GPT Image 1"""
        return f"""Create an appetizing marketing poster for a food product with these specifications:

1. PRODUCT PRESENTATION:
   - Center the food item as the focal point
   - Remove the original background completely
   - Enhance food textures and colors for appetite appeal
   - Add steam effects for hot items or fresh appearance for cold items
   - Ensure the food looks perfectly prepared

2. BRANDING ELEMENTS:
   - Company: "{product_details.get('company_name', '')}" at top in clean font
   - Product: "{product_details.get('product_name', '')}" below the food in bold font
   - Price: "{product_details.get('price', '')}" in an eye-catching badge
   - Tagline: "{product_details.get('tagline', '')}" emphasizing flavor/freshness
   - Location: "{product_details.get('address', '')}" at bottom in readable font

3. MARKETING ENHANCEMENT:
   - Background complementing the food's colors
   - Subtle ingredients in background for context
   - Professional food photography style
   - Balance all elements for premium restaurant look

Create a professional food marketing poster as described above.
"""

###################
# IMAGE GENERATION
###################

class ImageGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logger.info("ImageGenerator initialized with OpenAI API")
    
    def generate_marketing_image(self, product_image_path: str, product_details: Dict, product_type: str = "beverage") -> Optional[str]:
        """Generate a marketing image using GPT Image 1 API"""
        try:
            logger.info(f"Generating marketing image for {product_details.get('product_name', 'product')}")
            
            # Select the appropriate prompt template
            if product_type.lower() == "beverage":
                prompt = PromptTemplates.get_beverage_template(product_details)
            elif product_type.lower() == "food":
                prompt = PromptTemplates.get_food_template(product_details)
            else:
                prompt = PromptTemplates.get_master_template(product_details)
            
            # Open and read the product image
            with open(product_image_path, "rb") as img_file:
                # Use the edit endpoint to combine the product image with our marketing prompt
                result = self.client.images.edit(
                    model="gpt-image-1",
                    image=img_file,
                    prompt=prompt,
                    size="1024x1024",
                    quality="low"  # Changed to low for testing
                )
            
            # If no image in result, generate from scratch with the prompt
            if not result.data or not result.data[0].b64_json:
                logger.warning("No image returned from edit endpoint, trying generation endpoint")
                result = self.client.images.generate(
                    model="gpt-image-1",
                    prompt=prompt,
                    size="1024x1024",
                    quality="low"  # Changed to low for testing
                )
            
            # Get the generated image
            if result.data and result.data[0].b64_json:
                image_base64 = result.data[0].b64_json
                
                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                product_name_safe = product_details.get('product_name', 'product').replace(' ', '_')[:20]
                output_filename = f"{product_name_safe}_{timestamp}.png"
                output_path = os.path.join("images/output", output_filename)
                
                # Convert base64 to image
                image_bytes = base64.b64decode(image_base64)
                image = Image.open(BytesIO(image_bytes))
                
                # Optionally resize for optimization
                if image.size[0] > 1500 or image.size[1] > 1500:
                    image.thumbnail((1500, 1500), Image.LANCZOS)
                
                # Save the image
                image.save(output_path, format="PNG", optimize=True)
                
                logger.info(f"Marketing image saved to {output_path}")
                return output_path
            else:
                logger.error("No image data in API response")
                return None
                
        except Exception as e:
            logger.error(f"Error generating marketing image: {str(e)}")
            return None

###################
# WHATSAPP API
###################

class WhatsAppAPI:
    def __init__(self, access_token: str, phone_number_id: str):
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.api_url = f"{WHATSAPP_API_URL}/{phone_number_id}"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    
    def send_message(self, to: str, message: str) -> bool:
        """Send a text message via WhatsApp API"""
        try:
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": "text",
                "text": {
                    "preview_url": False,
                    "body": message
                }
            }
            
            response = requests.post(
                f"{self.api_url}/messages",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                logger.info(f"Message sent successfully to {to}")
                return True
            else:
                logger.error(f"Failed to send message: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {str(e)}")
            return False
    
    def send_image(self, to: str, image_url: str, caption: str = "") -> bool:
        """Send an image via WhatsApp API"""
        try:
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": "image",
                "image": {
                    "link": image_url,
                    "caption": caption
                }
            }
            
            response = requests.post(
                f"{self.api_url}/messages",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                logger.info(f"Image sent successfully to {to}")
                return True
            else:
                logger.error(f"Failed to send image: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending WhatsApp image: {str(e)}")
            return False
    
    def download_media(self, media_id: str) -> Optional[bytes]:
        """Download media from WhatsApp"""
        try:
            # First get the media URL
            url = f"{WHATSAPP_API_URL}/{media_id}"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to get media URL: {response.text}")
                return None
            
            media_url = response.json().get('url')
            if not media_url:
                logger.error("No media URL in response")
                return None
            
            # Download the actual media
            media_response = requests.get(media_url, headers=headers)
            if media_response.status_code == 200:
                return media_response.content
            else:
                logger.error(f"Failed to download media: {media_response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading media: {str(e)}")
            return None

###################
# MARKETING BOT
###################

class MarketingBot:
    def __init__(self, openai_key: str, whatsapp_token: str, whatsapp_phone_id: str):
        self.image_generator = ImageGenerator(openai_key)
        self.whatsapp_api = WhatsAppAPI(whatsapp_token, whatsapp_phone_id)
        logger.info("MarketingBot initialized with OpenAI and WhatsApp APIs")
    
    def process_request(self, user_id: str, product_image_path: str, product_details: Dict, product_type: str = "beverage") -> Dict:
        """Process a marketing image request"""
        try:
            logger.info(f"Processing request for user {user_id}")
            
            # Generate marketing image
            output_path = self.image_generator.generate_marketing_image(
                product_image_path,
                product_details,
                product_type
            )
            
            if output_path:
                # Get the Railway app URL for serving images
                app_url = os.getenv('RAILWAY_STATIC_URL', os.getenv('IMAGE_URL', 'https://fashioncore-ws-production.up.railway.app'))
                image_url = f"{app_url}/images/output/{os.path.basename(output_path)}"
                
                return {
                    "success": True,
                    "image_path": output_path,
                    "image_url": image_url
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to generate marketing image"
                }
                
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# Initialize marketing bot
marketing_bot = MarketingBot(
    openai_key=OPENAI_API_KEY,
    whatsapp_token=WHATSAPP_ACCESS_TOKEN,
    whatsapp_phone_id=WHATSAPP_PHONE_NUMBER_ID
)

###################
# FLASK ROUTES
###################

@app.route('/')
def home():
    """Render the home page"""
    return jsonify({
        "service": "Marketing Image Generator",
        "status": "running",
        "version": "1.0",
        "endpoint": "/webhook"
    })

@app.route('/health')
def health():
    """Health check endpoint for Railway"""
    return jsonify({"status": "healthy"})

@app.route('/images/<path:path>')
def serve_images(path):
    """Serve images from the images directory"""
    directory, filename = os.path.split(path)
    return send_from_directory(os.path.join('images', directory), filename)

# WhatsApp webhook verification endpoint
@app.route('/webhook', methods=['GET'])
def verify_webhook():
    """Verify WhatsApp webhook"""
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    
    if mode and token:
        if mode == 'subscribe' and token == WHATSAPP_VERIFY_TOKEN:
            logger.info("Webhook verified successfully")
            return challenge, 200
        else:
            logger.warning("Webhook verification failed")
            return 'Forbidden', 403
    
    return 'Bad Request', 400

# WhatsApp webhook message handler
@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages"""
    try:
        data = request.get_json()
        logger.info(f"Received webhook data: {json.dumps(data)}")
        
        if data.get('object') == 'whatsapp_business_account':
            for entry in data.get('entry', []):
                for change in entry.get('changes', []):
                    value = change.get('value', {})
                    messages = value.get('messages', [])
                    
                    for message in messages:
                        from_number = message.get('from')
                        message_type = message.get('type')
                        
                        # Handle text messages
                        if message_type == 'text':
                            text = message.get('text', {}).get('body', '')
                            handle_whatsapp_text(from_number, text)
                        
                        # Handle image messages
                        elif message_type == 'image':
                            image_id = message.get('image', {}).get('id')
                            handle_whatsapp_image(from_number, image_id)
        
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return jsonify({"status": "error"}), 500

def handle_whatsapp_text(from_number: str, text: str):
    """Handle incoming text messages from WhatsApp"""
    try:
        # Create session if not exists
        if from_number not in user_sessions:
            user_sessions[from_number] = {
                "product_image": None,
                "details": {},
                "state": "waiting_for_command"  # Track conversation state
            }
        
        session = user_sessions[from_number]
        
        # Check for start command
        if text.lower() == 'edit':
            session['state'] = 'waiting_for_image'
            session['product_image'] = None
            session['details'] = {}
            marketing_bot.whatsapp_api.send_message(
                from_number,
                "Welcome to Marketing Image Editor! 📸\n\n"
                "Please send your product image to begin.\n\n"
                "After sending the image, I'll ask for details like company name, product name, price, etc."
            )
            return
        
        # Check for generate command
        if text.lower() == 'generate':
            # Validate we have all required info
            if not session.get('product_image'):
                marketing_bot.whatsapp_api.send_message(
                    from_number,
                    "Please send a product image first.\n"
                    "To start, send 'edit'."
                )
                return
            
            details = session.get('details', {})
            if not all([details.get('company_name'), details.get('product_name'), details.get('price')]):
                missing = []
                if not details.get('company_name'):
                    missing.append('company name')
                if not details.get('product_name'):
                    missing.append('product name')
                if not details.get('price'):
                    missing.append('price')
                
                marketing_bot.whatsapp_api.send_message(
                    from_number,
                    f"Missing required details: {', '.join(missing)}\n\n"
                    "Please provide all required information."
                )
                return
            
            # Generate the image
            marketing_bot.whatsapp_api.send_message(from_number, "✨ Generating your marketing image...")
            
            result = marketing_bot.process_request(
                from_number,
                session['product_image'],
                details
            )
            
            if result['success']:
                # Send the generated image
                marketing_bot.whatsapp_api.send_image(
                    from_number,
                    result['image_url'],
                    "🎉 Here's your marketing image!\n\n"
                    "To create another image, send 'edit' again."
                )
                # Reset state
                session['state'] = 'waiting_for_command'
            else:
                marketing_bot.whatsapp_api.send_message(
                    from_number,
                    f"Sorry, I couldn't generate the image: {result.get('error', 'Unknown error')}\n\n"
                    "Please try again or send 'edit' to start over."
                )
            return
        
        # Handle conversation flow based on state
        if session['state'] == 'waiting_for_image':
            marketing_bot.whatsapp_api.send_message(
                from_number,
                "Please send your product image first.\n"
                "I'm waiting for an image file."
            )
            return
        
        elif session['state'] == 'waiting_for_details':
            # Parse the details
            lines = text.split('\n')
            detail_provided = False
            
            # Check if this is a structured message
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if 'company' in key:
                        session['details']['company_name'] = value
                        detail_provided = True
                    elif 'product' in key:
                        session['details']['product_name'] = value
                        detail_provided = True
                    elif 'price' in key:
                        session['details']['price'] = value
                        detail_provided = True
                    elif 'tagline' in key:
                        session['details']['tagline'] = value
                        detail_provided = True
                    elif 'address' in key or 'location' in key:
                        session['details']['address'] = value
                        detail_provided = True
            
            # If no structured details, try to interpret single line inputs
            if not detail_provided:
                if not session['details'].get('company_name'):
                    session['details']['company_name'] = text
                elif not session['details'].get('product_name'):
                    session['details']['product_name'] = text
                elif not session['details'].get('price'):
                    session['details']['price'] = text
                elif not session['details'].get('tagline'):
                    session['details']['tagline'] = text
                elif not session['details'].get('address'):
                    session['details']['address'] = text
            
            # Send updated status and next step
            status_msg = "📝 Current details:\n\n"
            status_msg += f"Company: {session['details'].get('company_name', '❌')}\n"
            status_msg += f"Product: {session['details'].get('product_name', '❌')}\n"
            status_msg += f"Price: {session['details'].get('price', '❌')}\n"
            status_msg += f"Tagline: {session['details'].get('tagline', '➖')}\n"
            status_msg += f"Address: {session['details'].get('address', '➖')}\n\n"
            
            # Check what's still needed
            if not session['details'].get('company_name'):
                status_msg += "👉 Please send your company name.\n"
            elif not session['details'].get('product_name'):
                status_msg += "👉 Please send your product name.\n"
            elif not session['details'].get('price'):
                status_msg += "👉 Please send the price.\n"
            else:
                status_msg += "✅ All required information received!\n\n"
                status_msg += "To generate the marketing image, send 'generate'\n"
                status_msg += "To add optional details (tagline, address), just send them."
            
            marketing_bot.whatsapp_api.send_message(from_number, status_msg)
            return
        
        # Default state - waiting for command
        else:
            marketing_bot.whatsapp_api.send_message(
                from_number,
                "👋 Welcome to Marketing Image Generator!\n\n"
                "To create a marketing image, send 'edit' to start."
            )
            
    except Exception as e:
        logger.error(f"Error handling WhatsApp text: {str(e)}")
        marketing_bot.whatsapp_api.send_message(
            from_number,
            "Sorry, an error occurred. Please try again.\n"
            "Send 'edit' to start over."
        )

def handle_whatsapp_image(from_number: str, image_id: str):
    """Handle incoming image messages from WhatsApp"""
    try:
        # Create session if not exists
        if from_number not in user_sessions:
            user_sessions[from_number] = {
                "product_image": None,
                "details": {},
                "state": "waiting_for_command"
            }
        
        session = user_sessions[from_number]
        
        # Check if we're in the right state to receive an image
        if session['state'] != 'waiting_for_image':
            marketing_bot.whatsapp_api.send_message(
                from_number,
                "I wasn't expecting an image right now.\n"
                "To start the process, please send 'edit' first."
            )
            return
        
        # Download the image from WhatsApp
        image_data = marketing_bot.whatsapp_api.download_media(image_id)
        
        if image_data:
            # Save the image
            filename = f"whatsapp_image_{from_number}_{int(datetime.now().timestamp())}.jpg"
            image_path = os.path.join("images/input", filename)
            
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            session['product_image'] = image_path
            session['state'] = 'waiting_for_details'
            
            marketing_bot.whatsapp_api.send_message(
                from_number,
                "✅ Product image received!\n\n"
                "Now please provide the following details:\n\n"
                "1️⃣ Company name\n"
                "2️⃣ Product name\n"
                "3️⃣ Price\n"
                "4️⃣ Tagline (optional)\n"
                "5️⃣ Address (optional)\n\n"
                "You can send them one by one or all at once.\n"
                "Example format:\n"
                "Company: ABC Corp\n"
                "Product: Premium Coffee\n"
                "Price: $20\n\n"
                "When you're ready to generate the image, send 'generate'"
            )
        else:
            marketing_bot.whatsapp_api.send_message(
                from_number,
                "Sorry, I couldn't download your image. Please try again.\n"
                "Send 'edit' to start over."
            )
            
    except Exception as e:
        logger.error(f"Error handling WhatsApp image: {str(e)}")
        marketing_bot.whatsapp_api.send_message(
            from_number,
            "Sorry, I couldn't process your image. Please try again.\n"
            "Start over by sending 'edit'."
        )

###################
# MAIN APPLICATION
###################

if __name__ == '__main__':
    # Get port from environment variable (Railway provides this)
    port = int(os.getenv('PORT', 5000))
    
    # Run the Flask app
    logger.info(f"Starting Marketing Bot API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False for production