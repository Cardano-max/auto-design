#!/usr/bin/env python3
"""
Marketing Image Generator - Twilio SMS Version
--------------------------------------------
Production version with Twilio SMS API integration
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
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

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
logger = logging.getLogger("MarketingBotTwilio")

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

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', 'ACfa677ef9401c892d4480adb9875ea361')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '25badeb30fa8dcda00d49a941e111a40')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER', '+19787344086')

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

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
# TWILIO SMS/MMS API
###################

class TwilioAPI:
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number
        logger.info("TwilioAPI initialized successfully")
    
    def send_message(self, to: str, message: str) -> bool:
        """Send a text message via Twilio SMS"""
        try:
            message = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to
            )
            logger.info(f"Message sent successfully to {to}, SID: {message.sid}")
            return True
        except Exception as e:
            logger.error(f"Error sending Twilio message: {str(e)}")
            return False
    
    def send_mms(self, to: str, message: str, media_url: str) -> bool:
        """Send an MMS message via Twilio"""
        try:
            message = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to,
                media_url=[media_url]
            )
            logger.info(f"MMS sent successfully to {to}, SID: {message.sid}")
            return True
        except Exception as e:
            logger.error(f"Error sending Twilio MMS: {str(e)}")
            return False

###################
# MARKETING BOT
###################

class MarketingBot:
    def __init__(self, openai_key: str, twilio_account_sid: str, twilio_auth_token: str, twilio_from_number: str):
        self.image_generator = ImageGenerator(openai_key)
        self.twilio_api = TwilioAPI(twilio_account_sid, twilio_auth_token, twilio_from_number)
        logger.info("MarketingBot initialized with OpenAI and Twilio APIs")
    
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
                app_url = os.getenv('RAILWAY_STATIC_URL', os.getenv('APP_URL', 'https://auto-design-production.up.railway.app'))
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
    twilio_account_sid=TWILIO_ACCOUNT_SID,
    twilio_auth_token=TWILIO_AUTH_TOKEN,
    twilio_from_number=TWILIO_PHONE_NUMBER
)

###################
# FLASK ROUTES
###################

@app.route('/')
def home():
    """Render the home page"""
    return jsonify({
        "service": "Marketing Image Generator (Twilio Version)",
        "status": "running",
        "version": "1.0",
        "endpoint": "/sms"
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

# Twilio SMS webhook endpoint
@app.route('/sms', methods=['POST'])
def sms_webhook():
    """Handle incoming SMS messages"""
    try:
        # Get the message details
        from_number = request.form.get('From')
        message_body = request.form.get('Body', '').strip().lower()
        num_media = int(request.form.get('NumMedia', 0))
        
        logger.info(f"Received SMS from {from_number}: {message_body}, Media: {num_media}")
        
        # Create TwiML response
        resp = MessagingResponse()
        
        # Handle media messages (images)
        if num_media > 0:
            media_url = request.form.get('MediaUrl0')
            content_type = request.form.get('MediaContentType0')
            
            if content_type and content_type.startswith('image/'):
                handle_image_message(from_number, media_url, resp)
            else:
                resp.message("Please send an image file to create your marketing material.")
        else:
            # Handle text messages
            handle_text_message(from_number, message_body, resp)
        
        return str(resp)
    except Exception as e:
        logger.error(f"SMS webhook error: {str(e)}")
        resp = MessagingResponse()
        resp.message("Sorry, an error occurred. Please try again.")
        return str(resp)

def handle_text_message(from_number: str, text: str, resp: MessagingResponse):
    """Handle incoming text messages"""
    try:
        # Create session if not exists
        if from_number not in user_sessions:
            user_sessions[from_number] = {
                "product_image": None,
                "details": {},
                "state": "waiting_for_command"
            }
        
        session = user_sessions[from_number]
        
        # Check for start command
        if text == 'edit':
            session['state'] = 'waiting_for_image'
            session['product_image'] = None
            session['details'] = {}
            resp.message("Welcome to Marketing Image Editor! 📸\n\n"
                        "Please send your product image to begin.\n\n"
                        "After sending the image, I'll ask for details like company name, product name, price, etc.")
            return
        
        # Check for generate command
        if text == 'generate':
            # Validate we have all required info
            if not session.get('product_image'):
                resp.message("Please send a product image first.\n"
                           "To start, send 'edit'.")
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
                
                resp.message(f"Missing required details: {', '.join(missing)}\n\n"
                           "Please provide all required information.")
                return
            
            # Generate the image
            resp.message("✨ Generating your marketing image...")
            
            # Process in the background
            result = marketing_bot.process_request(
                from_number,
                session['product_image'],
                details
            )
            
            if result['success']:
                # Send the generated image
                marketing_bot.twilio_api.send_mms(
                    from_number,
                    "🎉 Here's your marketing image!\n\n"
                    "To create another image, send 'edit' again.",
                    result['image_url']
                )
                # Reset state
                session['state'] = 'waiting_for_command'
            else:
                marketing_bot.twilio_api.send_message(
                    from_number,
                    f"Sorry, I couldn't generate the image: {result.get('error', 'Unknown error')}\n\n"
                    "Please try again or send 'edit' to start over."
                )
            return
        
        # Handle conversation flow based on state
        if session['state'] == 'waiting_for_image':
            resp.message("Please send your product image first.\n"
                        "I'm waiting for an image file.")
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
            
            resp.message(status_msg)
            return
        
        # Default state - waiting for command
        else:
            resp.message("👋 Welcome to Marketing Image Generator!\n\n"
                        "To create a marketing image, send 'edit' to start.")
            
    except Exception as e:
        logger.error(f"Error handling text message: {str(e)}")
        resp.message("Sorry, an error occurred. Please try again.\n"
                    "Send 'edit' to start over.")

def handle_image_message(from_number: str, media_url: str, resp: MessagingResponse):
    """Handle incoming image messages"""
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
            resp.message("I wasn't expecting an image right now.\n"
                        "To start the process, please send 'edit' first.")
            return
        
        # Download the image from Twilio
        try:
            response = requests.get(media_url)
            if response.status_code == 200:
                # Save the image
                filename = f"twilio_image_{from_number.replace('+', '')}_{int(datetime.now().timestamp())}.jpg"
                image_path = os.path.join("images/input", filename)
                
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                session['product_image'] = image_path
                session['state'] = 'waiting_for_details'
                
                resp.message("✅ Product image received!\n\n"
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
                            "When you're ready to generate the image, send 'generate'")
            else:
                resp.message("Sorry, I couldn't download your image. Please try again.")
        except Exception as e:
            logger.error(f"Error downloading image: {str(e)}")
            resp.message("Sorry, I couldn't process your image. Please try again.")
            
    except Exception as e:
        logger.error(f"Error handling image message: {str(e)}")
        resp.message("Sorry, I couldn't process your image. Please try again.\n"
                    "Start over by sending 'edit'.")

###################
# MAIN APPLICATION
###################

if __name__ == '__main__':
    # Get port from environment variable (Railway provides this)
    port = int(os.getenv('PORT', 5000))
    
    # Run the Flask app
    logger.info(f"Starting Marketing Bot API (Twilio Version) on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False for production