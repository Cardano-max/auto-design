#!/usr/bin/env python3
"""
Marketing Image Generator - WaAPI WhatsApp Version
--------------------------------------------
Production version with WaAPI WhatsApp API integration
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
logger = logging.getLogger("MarketingBotWhatsApp")

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Create directories for storing images
os.makedirs('images/input', exist_ok=True)
os.makedirs('images/output', exist_ok=True)

# Store user sessions
user_sessions = {}

# Initialize OpenAI with compatibility handling
# IMPORTANT: Add OPENAI_API_KEY to your Railway variables too!
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Add this to Railway variables

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables!")
    raise ValueError("OPENAI_API_KEY is required")

try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    OPENAI_VERSION = "new"
except Exception as e:
    logger.warning(f"Using legacy OpenAI API: {str(e)}")
    import openai
    openai.api_key = OPENAI_API_KEY
    openai_client = openai
    OPENAI_VERSION = "legacy"

# WaAPI configuration - from environment variables
WAAPI_API_TOKEN = os.getenv('WAAPI_API_TOKEN')
WAAPI_INSTANCE_ID = os.getenv('WAAPI_INSTANCE_ID')

# Validate required environment variables
required_vars = {
    'WAAPI_API_TOKEN': WAAPI_API_TOKEN,
    'WAAPI_INSTANCE_ID': WAAPI_INSTANCE_ID
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

###################
# PROMPT TEMPLATES
###################

class PromptTemplates:
    @staticmethod
    def get_master_template(product_details: Dict) -> str:
        """Generate the master prompt template with product details"""
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
        """Generate beverage-specific prompt template"""
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
        """Generate food-specific prompt template"""
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
        self.api_key = api_key
        self.client = openai_client
        self.openai_version = OPENAI_VERSION
        logger.info(f"ImageGenerator initialized with OpenAI API ({self.openai_version} version)")
    
    def generate_marketing_image(self, product_image_path: str, product_details: Dict, product_type: str = "beverage") -> Optional[str]:
        """Generate a marketing image using OpenAI API"""
        try:
            logger.info(f"Generating marketing image for {product_details.get('product_name', 'product')}")
            
            # Select the appropriate prompt template
            if product_type.lower() == "beverage":
                prompt = PromptTemplates.get_beverage_template(product_details)
            elif product_type.lower() == "food":
                prompt = PromptTemplates.get_food_template(product_details)
            else:
                prompt = PromptTemplates.get_master_template(product_details)
            
            # Generate the image
            try:
                if self.openai_version == "new":
                    # Using newer OpenAI client
                    result = self.client.images.generate(
                        model="dall-e-3",
                        prompt=prompt,
                        size="1024x1024",
                        quality="standard",
                        n=1
                    )
                    
                    if hasattr(result.data[0], 'url'):
                        image_url = result.data[0].url
                        response = requests.get(image_url)
                        image_bytes = response.content
                    else:
                        image_bytes = base64.b64decode(result.data[0].b64_json)
                else:
                    # Using legacy OpenAI API
                    result = self.client.Image.create(
                        prompt=prompt,
                        n=1,
                        size="1024x1024"
                    )
                    
                    if 'data' in result and len(result['data']) > 0:
                        if 'url' in result['data'][0]:
                            image_url = result['data'][0]['url']
                            response = requests.get(image_url)
                            image_bytes = response.content
                        elif 'b64_json' in result['data'][0]:
                            image_bytes = base64.b64decode(result['data'][0]['b64_json'])
                    else:
                        logger.error("No image data in response")
                        return None
                    
                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                product_name_safe = product_details.get('product_name', 'product').replace(' ', '_')[:20]
                output_filename = f"{product_name_safe}_{timestamp}.png"
                output_path = os.path.join("images/output", output_filename)
                
                # Convert to image
                image = Image.open(BytesIO(image_bytes))
                
                # Optionally resize for optimization
                if image.size[0] > 1500 or image.size[1] > 1500:
                    image.thumbnail((1500, 1500), Image.LANCZOS)
                
                # Save the image
                image.save(output_path, format="PNG", optimize=True)
                
                logger.info(f"Marketing image saved to {output_path}")
                return output_path
                
            except Exception as api_error:
                logger.error(f"API Error: {str(api_error)}")
                
                # Fallback to simpler generation
                try:
                    logger.info("Trying fallback generation method")
                    if self.openai_version == "new":
                        result = self.client.images.generate(
                            model="dall-e-2",
                            prompt=prompt,
                            size="1024x1024",
                            n=1
                        )
                    else:
                        result = self.client.Image.create(
                            prompt=prompt,
                            n=1,
                            size="1024x1024"
                        )
                    
                    # Process the response (same as above)
                    image_bytes = None
                    if self.openai_version == "new":
                        if hasattr(result.data[0], 'url'):
                            response = requests.get(result.data[0].url)
                            image_bytes = response.content
                        else:
                            image_bytes = base64.b64decode(result.data[0].b64_json)
                    else:
                        if 'data' in result and len(result['data']) > 0:
                            if 'url' in result['data'][0]:
                                response = requests.get(result['data'][0]['url'])
                                image_bytes = response.content
                            elif 'b64_json' in result['data'][0]:
                                image_bytes = base64.b64decode(result['data'][0]['b64_json'])
                    
                    if not image_bytes:
                        return None
                    
                    # Save the image
                    image = Image.open(BytesIO(image_bytes))
                    if image.size[0] > 1500 or image.size[1] > 1500:
                        image.thumbnail((1500, 1500), Image.LANCZOS)
                    image.save(output_path, format="PNG", optimize=True)
                    
                    logger.info(f"Marketing image saved to {output_path} (fallback method)")
                    return output_path
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback generation failed: {str(fallback_error)}")
                    return None
                
        except Exception as e:
            logger.error(f"Error generating marketing image: {str(e)}")
            return None

###################
# WAAPI WHATSAPP API
###################

class WaAPIClient:
    def __init__(self, api_token: str, instance_id: str):
        self.api_token = api_token
        self.instance_id = instance_id
        self.api_base_url = "https://waapi.app/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        logger.info("WaAPIClient initialized successfully")
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make a request to the WaAPI API"""
        url = f"{self.api_base_url}/{endpoint}"
        try:
            if method.lower() == "get":
                response = requests.get(url, headers=self.headers)
            elif method.lower() == "post":
                response = requests.post(url, headers=self.headers, json=data)
            elif method.lower() == "put":
                response = requests.put(url, headers=self.headers, json=data)
            elif method.lower() == "delete":
                response = requests.delete(url, headers=self.headers)
            else:
                logger.error(f"Invalid method: {method}")
                return {"success": False, "error": "Invalid method"}
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error making request to WaAPI: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_instance_status(self) -> Dict:
        """Get the status of the instance"""
        return self._make_request("get", f"instances/{self.instance_id}/client/status")
    
    def send_message(self, to: str, message: str) -> Dict:
        """Send a text message via WhatsApp"""
        # Format phone number to WhatsApp format if needed
        to = self._format_phone_number(to)
        
        data = {
            "chatId": to,
            "message": message
        }
        
        result = self._make_request(
            "post", 
            f"instances/{self.instance_id}/client/action/send-message", 
            data
        )
        
        if result.get("status") == "success":
            logger.info(f"Message sent successfully to {to}")
            return {"success": True, "data": result.get("data", {})}
        else:
            logger.error(f"Error sending message: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get("error", "Unknown error")}
    
    def send_media(self, to: str, caption: str = "", media_url: str = None, media_base64: str = None, 
                  filename: str = None, is_sticker: bool = False) -> Dict:
        """Send a media message via WhatsApp"""
        # Format phone number to WhatsApp format if needed
        to = self._format_phone_number(to)
        
        data = {
            "chatId": to,
            "mediaCaption": caption
        }
        
        # Add either URL or base64 data
        if media_url:
            data["mediaUrl"] = media_url
        elif media_base64:
            data["mediaBase64"] = media_base64
            # Filename is required for base64 media
            data["mediaName"] = filename or f"media_{int(datetime.now().timestamp())}.png"
        
        # Set if image should be sent as sticker
        if is_sticker:
            data["asSticker"] = True
            
        result = self._make_request(
            "post", 
            f"instances/{self.instance_id}/client/action/send-media", 
            data
        )
        
        if result.get("status") == "success":
            logger.info(f"Media sent successfully to {to}")
            return {"success": True, "data": result.get("data", {})}
        else:
            logger.error(f"Error sending media: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get("error", "Unknown error")}
    
    def _format_phone_number(self, phone_number: str) -> str:
        """Format phone number to the expected format for WhatsApp"""
        # Remove any non-numeric characters except the @ if it's already formatted
        if '@c.us' in phone_number:
            return phone_number
            
        # Strip any non-numeric characters (like +, spaces, etc.)
        clean_number = ''.join(filter(str.isdigit, phone_number))
        
        # Format for WhatsApp: number@c.us
        return f"{clean_number}@c.us"
    
    def is_registered_user(self, phone_number: str) -> bool:
        """Check if a phone number is registered on WhatsApp"""
        # Format the phone number
        contact_id = self._format_phone_number(phone_number)
        
        data = {
            "contactId": contact_id
        }
        
        result = self._make_request(
            "post", 
            f"instances/{self.instance_id}/client/action/is-registered-user", 
            data
        )
        
        if result.get("status") == "success" and "data" in result:
            return result["data"].get("isRegisteredUser", False)
        
        return False

###################
# MARKETING BOT
###################

class MarketingBot:
    def __init__(self, openai_key: str, waapi_token: str, waapi_instance_id: str):
        self.image_generator = ImageGenerator(openai_key)
        self.waapi_client = WaAPIClient(waapi_token, waapi_instance_id)
        logger.info("MarketingBot initialized with OpenAI and WaAPI")
    
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
    waapi_token=WAAPI_API_TOKEN,
    waapi_instance_id=WAAPI_INSTANCE_ID
)

###################
# FLASK ROUTES
###################

@app.route('/')
def home():
    """Render the home page"""
    return jsonify({
        "service": "Marketing Image Generator (WhatsApp Version)",
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

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages via WaAPI webhook"""
    try:
        # Extract the webhook data
        webhook_data = request.json
        logger.info(f"Received webhook: {json.dumps(webhook_data)}")
        
        # Check if this is a message event
        if webhook_data.get('event') == 'message':
            message_data = webhook_data.get('data', {}).get('message', {})
            
            # Extract message details
            message_type = message_data.get('type')
            from_number = message_data.get('from')
            body = message_data.get('body', '').strip().lower()
            has_media = message_data.get('hasMedia', False)
            
            logger.info(f"Received message from {from_number}: {body}, Media: {has_media}")
            
            # Check if the number is in the proper format
            if '@c.us' not in from_number:
                logger.error(f"Invalid phone number format: {from_number}")
                return jsonify({"status": "error", "message": "Invalid phone number format"})
            
            # Handle media messages (images)
            if has_media and message_type in ['image', 'sticker']:
                media_data = webhook_data.get('data', {}).get('media', {})
                handle_image_message(from_number, media_data)
            else:
                # Handle text messages
                handle_text_message(from_number, body)
            
            return jsonify({"status": "success"})
        
        # Handle other event types if needed
        return jsonify({"status": "success", "message": "Event processed"})
    
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

def handle_text_message(from_number: str, text: str):
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
            marketing_bot.waapi_client.send_message(
                from_number,
                "Welcome to Marketing Image Editor! 📸\n\n"
                "Please send your product image to begin.\n\n"
                "After sending the image, I'll ask for details like company name, product name, price, etc."
            )
            return
        
        # Check for generate command
        if text == 'generate':
            # Validate we have all required info
            if not session.get('product_image'):
                marketing_bot.waapi_client.send_message(
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
                
                marketing_bot.waapi_client.send_message(
                    from_number,
                    f"Missing required details: {', '.join(missing)}\n\n"
                    "Please provide all required information."
                )
                return
            
            # Generate the image
            marketing_bot.waapi_client.send_message(
                from_number,
                "✨ Generating your marketing image..."
            )
            
            # Process in the background
            result = marketing_bot.process_request(
                from_number,
                session['product_image'],
                details
            )
            
            if result['success']:
                # Get local file path
                image_path = result['image_path']
                
                # Read the image for base64 encoding
                with open(image_path, 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Send the generated image
                marketing_bot.waapi_client.send_media(
                    from_number,
                    "🎉 Here's your marketing image!\n\n"
                    "To create another image, send 'edit' again.",
                    media_base64=img_base64,
                    filename=os.path.basename(image_path)
                )
                # Reset state
                session['state'] = 'waiting_for_command'
            else:
                marketing_bot.waapi_client.send_message(
                    from_number,
                    f"Sorry, I couldn't generate the image: {result.get('error', 'Unknown error')}\n\n"
                    "Please try again or send 'edit' to start over."
                )
            return
        
        # Handle conversation flow based on state
        if session['state'] == 'waiting_for_image':
            marketing_bot.waapi_client.send_message(
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
            
            marketing_bot.waapi_client.send_message(from_number, status_msg)
            return
        
        # Default state - waiting for command
        else:
            marketing_bot.waapi_client.send_message(
                from_number,
                "👋 Welcome to Marketing Image Generator!\n\n"
                "To create a marketing image, send 'edit' to start."
            )
            
    except Exception as e:
        logger.error(f"Error handling text message: {str(e)}")
        marketing_bot.waapi_client.send_message(
            from_number,
            "Sorry, an error occurred. Please try again.\n"
            "Send 'edit' to start over."
        )

def handle_image_message(from_number: str, media_data):
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
            marketing_bot.waapi_client.send_message(
                from_number,
                "I wasn't expecting an image right now.\n"
                "To start the process, please send 'edit' first."
            )
            return
        
        # Process the image from media_data (base64 encoded)
        try:
            if media_data.get('data'):  # Check if we have base64 data
                # Decode base64 data
                image_bytes = base64.b64decode(media_data['data'])
                
                # Save the image
                filename = f"whatsapp_image_{from_number.replace('@c.us', '')}_{int(datetime.now().timestamp())}.jpg"
                image_path = os.path.join("images/input", filename)
                
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
                session['product_image'] = image_path
                session['state'] = 'waiting_for_details'
                
                marketing_bot.waapi_client.send_message(
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
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "Sorry, I couldn't process your image. Please try again."
                )
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            marketing_bot.waapi_client.send_message(
                from_number,
                "Sorry, I couldn't process your image. Please try again."
            )
            
    except Exception as e:
        logger.error(f"Error handling image message: {str(e)}")
        marketing_bot.waapi_client.send_message(
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
    logger.info(f"Starting Marketing Bot API (WhatsApp Version) on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False for production