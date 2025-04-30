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
import time
from datetime import datetime
from typing import Optional, Dict, List, Any
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure production-grade logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
log_path = os.getenv('LOG_PATH', 'logs')
os.makedirs(log_path, exist_ok=True)

# Create a rotating file handler
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            os.path.join(log_path, "marketing_bot.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MarketingBotWhatsApp")
logger.info(f"Logging initialized at {log_level} level")

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Create directories for storing images
os.makedirs('images/input', exist_ok=True)
os.makedirs('images/output', exist_ok=True)

# Store user sessions with expiration management
user_sessions = {}
SESSION_TIMEOUT_HOURS = 24  # Auto-expire sessions after 24 hours of inactivity

# Store processed messages to prevent duplicates
processed_messages = {}
MAX_PROCESSED_MESSAGES = 1000  # Limit the cache size

# Store last message time for rate limiting
last_message_time = {}
RATE_LIMIT_SECONDS = 2  # Minimum seconds between messages

# Initialize OpenAI with API key from environment variables
# CRITICAL: Set this in Railway environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables!")
    raise ValueError("OPENAI_API_KEY is required")

try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    OPENAI_VERSION = "new"
            logger.info("Using new OpenAI client version")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}", exc_info=True)
    raise ValueError("OpenAI client initialization failed. Please check your API key and dependencies.")

# WaAPI configuration - from environment variables
WAAPI_API_TOKEN = os.getenv('WAAPI_API_TOKEN')
WAAPI_INSTANCE_ID = os.getenv('WAAPI_INSTANCE_ID')
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET')

# Validate required environment variables
required_vars = {
    'WAAPI_API_TOKEN': WAAPI_API_TOKEN,
    'WAAPI_INSTANCE_ID': WAAPI_INSTANCE_ID
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

logger.info(f"Environment variables loaded. Instance ID: {WAAPI_INSTANCE_ID}")

###################
# PROMPT TEMPLATES
###################

class PromptTemplates:
    @staticmethod
    def get_master_template(product_details: Dict) -> str:
        """Generate the master prompt template with product details"""
        logger.debug(f"Generating master template with product details: {product_details}")
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
   - Background: light gradient that complements the product's colors
   - Use complementary colors that enhance the product's appeal
   - Ensure all text is perfectly readable

The final image should look like it was created by an expert graphic designer for a premium brand's marketing campaign.
"""

    @staticmethod
    def get_beverage_template(product_details: Dict) -> str:
        """Generate beverage-specific prompt template"""
        logger.debug(f"Generating beverage template with product details: {product_details}")
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
        logger.debug(f"Generating food template with product details: {product_details}")
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
        logger.info("Initializing ImageGenerator")
        self.api_key = api_key
        self.client = openai_client
        self.openai_version = OPENAI_VERSION
        logger.info(f"ImageGenerator initialized with OpenAI API ({self.openai_version} version)")
    
    def generate_marketing_image(self, product_image_path: str, product_details: Dict, product_type: str = "beverage") -> Optional[str]:
        """Generate a marketing image using OpenAI API"""
        try:
            product_name = product_details.get('product_name', 'product')
            logger.info(f"Starting image generation for {product_name} as {product_type}")
            
            # Select the appropriate prompt template
            if product_type.lower() == "beverage":
                prompt = PromptTemplates.get_beverage_template(product_details)
                logger.info("Using beverage template")
            elif product_type.lower() == "food":
                prompt = PromptTemplates.get_food_template(product_details)
                logger.info("Using food template")
            else:
                prompt = PromptTemplates.get_master_template(product_details)
                logger.info("Using master template")
            
            # Log the full prompt for debugging
            logger.info(f"Generated prompt: {prompt}")
            
            # Generate the image
            try:
                logger.info("Sending image generation request to OpenAI API")
                if self.openai_version == "new":
                    # Using newer OpenAI client with gpt-image-1 model
                    logger.info("Using new OpenAI client for image generation with gpt-image-1 model")
                    # Log the exact API parameters for debugging
                    logger.debug(f"OpenAI API parameters: model=gpt-image-1, size=1024x1024, quality=low")
                    
                    # Add retries for production reliability
                    max_retries = 3
                    retry_delay = 2
                    
                    for retry in range(max_retries):
                        try:
                            result = self.client.images.generate(
                                model="gpt-image-1",
                                prompt=prompt,
                                size="1024x1024",
                                n=1,
                                response_format="b64_json",
                                quality="low"  # Use low quality for production to save credits
                            )
                            # If successful, break the retry loop
                            break
                        except Exception as retry_error:
                            logger.warning(f"API call attempt {retry+1} failed: {str(retry_error)}")
                            if retry < max_retries - 1:
                                logger.info(f"Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                # Last attempt failed, re-raise the exception
                                logger.error("All API call retries failed")
                                raise
                    logger.info("OpenAI API request completed")
                    
                    if hasattr(result, 'data') and len(result.data) > 0 and hasattr(result.data[0], 'b64_json'):
                        logger.info("Image base64 data received, decoding")
                        image_bytes = base64.b64decode(result.data[0].b64_json)
                        logger.info("Image decoded successfully")
                    else:
                        logger.error("No image data in OpenAI response")
                        return None
                else:
                    # Using legacy OpenAI API
                    logger.info("Using legacy OpenAI client for image generation")
                    result = self.client.Image.create(
                        prompt=prompt,
                        n=1,
                        size="1024x1024",
                        response_format="b64_json"
                    )
                    logger.info("OpenAI API request completed")
                    
                    if 'data' in result and len(result['data']) > 0:
                        if 'b64_json' in result['data'][0]:
                            logger.info("Image base64 data received, decoding")
                            image_bytes = base64.b64decode(result['data'][0]['b64_json'])
                            logger.info("Image decoded successfully")
                        else:
                            logger.error("No base64 data in OpenAI response")
                            return None
                    else:
                        logger.error("No image data in response")
                        return None
                    
                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                product_name_safe = product_details.get('product_name', 'product').replace(' ', '_')[:20]
                output_filename = f"{product_name_safe}_{timestamp}.png"
                output_path = os.path.join("images/output", output_filename)
                
                # Convert to image
                logger.info("Creating image from bytes")
                image = Image.open(BytesIO(image_bytes))
                
                # Optionally resize for optimization
                if image.size[0] > 1500 or image.size[1] > 1500:
                    logger.info(f"Resizing image from {image.size} to max 1500px")
                    image.thumbnail((1500, 1500), Image.LANCZOS)
                
                # Save the image
                logger.info(f"Saving image to {output_path}")
                image.save(output_path, format="PNG", optimize=True)
                
                logger.info(f"Marketing image saved to {output_path}")
                return output_path
                
            except Exception as api_error:
                logger.error(f"OpenAI API Error: {str(api_error)}", exc_info=True)
                
                # Fallback to dall-e-2 model
                try:
                    logger.info("Trying fallback generation with DALL-E 2")
                    if self.openai_version == "new":
                        logger.info("Falling back to DALL-E 2 with new client")
                        result = self.client.images.generate(
                            model="dall-e-2",
                            prompt=prompt,
                            size="1024x1024",
                            n=1,
                            response_format="b64_json"
                        )
                    else:
                        logger.info("Retrying with legacy client")
                        result = self.client.Image.create(
                            prompt=prompt,
                            n=1,
                            size="1024x1024",
                            response_format="b64_json"
                        )
                    
                    # Process the response
                    image_bytes = None
                    if self.openai_version == "new":
                        logger.info("Processing fallback response from new client")
                        if hasattr(result, 'data') and len(result.data) > 0 and hasattr(result.data[0], 'b64_json'):
                            logger.info("Fallback: Image base64 data received, decoding")
                            image_bytes = base64.b64decode(result.data[0].b64_json)
                    else:
                        logger.info("Processing fallback response from legacy client")
                        if 'data' in result and len(result['data']) > 0 and 'b64_json' in result['data'][0]:
                            logger.info("Fallback: Image base64 data received, decoding")
                            image_bytes = base64.b64decode(result['data'][0]['b64_json'])
                    
                    if not image_bytes:
                        logger.error("Fallback generation failed: No image data")
                        return None
                    
                    # Save the image
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    product_name_safe = product_details.get('product_name', 'product').replace(' ', '_')[:20]
                    output_filename = f"{product_name_safe}_{timestamp}.png"
                    output_path = os.path.join("images/output", output_filename)
                    
                    logger.info("Saving fallback image")
                    image = Image.open(BytesIO(image_bytes))
                    if image.size[0] > 1500 or image.size[1] > 1500:
                        logger.info(f"Resizing fallback image from {image.size} to max 1500px")
                        image.thumbnail((1500, 1500), Image.LANCZOS)
                    image.save(output_path, format="PNG", optimize=True)
                    
                    logger.info(f"Marketing image saved to {output_path} (fallback method)")
                    return output_path
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback generation failed: {str(fallback_error)}", exc_info=True)
                    return None
                
        except Exception as e:
            logger.error(f"Error generating marketing image: {str(e)}", exc_info=True)
            return None

###################
# WAAPI WHATSAPP API
###################

class WaAPIClient:
    def __init__(self, api_token: str, instance_id: str):
        logger.info(f"Initializing WaAPIClient for instance {instance_id}")
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
        logger.info(f"Making {method} request to {endpoint}")
        try:
            if method.lower() == "get":
                response = requests.get(url, headers=self.headers)
            elif method.lower() == "post":
                logger.debug(f"POST data: {json.dumps(data)}")
                response = requests.post(url, headers=self.headers, json=data)
            elif method.lower() == "put":
                response = requests.put(url, headers=self.headers, json=data)
            elif method.lower() == "delete":
                response = requests.delete(url, headers=self.headers)
            else:
                logger.error(f"Invalid method: {method}")
                return {"success": False, "error": "Invalid method"}
            
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Response: {json.dumps(result)}")
            return result
        except Exception as e:
            logger.error(f"Error making request to WaAPI: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_instance_status(self) -> Dict:
        """Get the status of the instance"""
        logger.info("Checking instance status")
        return self._make_request("get", f"instances/{self.instance_id}/client/status")
    
    def send_message(self, to: str, message: str) -> Dict:
        """Send a text message via WhatsApp"""
        # Format phone number to WhatsApp format if needed
        to = self._format_phone_number(to)
        
        # Apply rate limiting
        current_time = time.time()
        if to in last_message_time and current_time - last_message_time[to] < 2:
            logger.info(f"Rate limiting message to {to} - too soon after last message")
            return {"success": False, "error": "Rate limited"}
        
        data = {
            "chatId": to,
            "message": message
        }
        
        logger.info(f"Sending message to {to}: {message[:50]}{'...' if len(message) > 50 else ''}")
        result = self._make_request(
            "post", 
            f"instances/{self.instance_id}/client/action/send-message", 
            data
        )
        
        if result.get("status") == "success":
            logger.info(f"Message sent successfully to {to}")
            # Update rate limiting
            last_message_time[to] = current_time
            return {"success": True, "data": result.get("data", {})}
        else:
            logger.error(f"Error sending message: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get("error", "Unknown error")}
    
    def send_media(self, to: str, caption: str = "", media_url: str = None, media_base64: str = None, 
                  filename: str = None, is_sticker: bool = False) -> Dict:
        """Send a media message via WhatsApp"""
        # Format phone number to WhatsApp format if needed
        to = self._format_phone_number(to)
        
        # Apply rate limiting
        current_time = time.time()
        if to in last_message_time and current_time - last_message_time[to] < 2:
            logger.info(f"Rate limiting media message to {to} - too soon after last message")
            return {"success": False, "error": "Rate limited"}
        
        data = {
            "chatId": to,
            "mediaCaption": caption
        }
        
        # Add either URL or base64 data
        if media_url:
            logger.info(f"Sending media from URL to {to}")
            data["mediaUrl"] = media_url
        elif media_base64:
            logger.info(f"Sending media from base64 to {to}")
            data["mediaBase64"] = media_base64
            # Filename is required for base64 media
            data["mediaName"] = filename or f"media_{int(datetime.now().timestamp())}.png"
        else:
            logger.error("No media URL or base64 data provided")
            return {"success": False, "error": "No media provided"}
        
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
            # Update rate limiting
            last_message_time[to] = current_time
            return {"success": True, "data": result.get("data", {})}
        else:
            logger.error(f"Error sending media: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get("error", "Unknown error")}
    
    def _format_phone_number(self, phone_number: str) -> str:
        """Format phone number to the expected format for WhatsApp"""
        logger.debug(f"Formatting phone number: {phone_number}")
        # Remove any non-numeric characters except the @ if it's already formatted
        if '@c.us' in phone_number:
            return phone_number
            
        # Strip any non-numeric characters (like +, spaces, etc.)
        clean_number = ''.join(filter(str.isdigit, phone_number))
        
        # Format for WhatsApp: number@c.us
        result = f"{clean_number}@c.us"
        logger.debug(f"Formatted phone number: {result}")
        return result
    
    def is_registered_user(self, phone_number: str) -> bool:
        """Check if a phone number is registered on WhatsApp"""
        # Format the phone number
        contact_id = self._format_phone_number(phone_number)
        logger.info(f"Checking if {contact_id} is registered on WhatsApp")
        
        data = {
            "contactId": contact_id
        }
        
        result = self._make_request(
            "post", 
            f"instances/{self.instance_id}/client/action/is-registered-user", 
            data
        )
        
        if result.get("status") == "success" and "data" in result:
            is_registered = result["data"].get("isRegisteredUser", False)
            logger.info(f"User {contact_id} is registered: {is_registered}")
            return is_registered
        
        logger.error(f"Failed to check if user is registered: {result.get('error', 'Unknown error')}")
        return False

###################
# MARKETING BOT
###################

class MarketingBot:
    def __init__(self, openai_key: str, waapi_token: str, waapi_instance_id: str):
        logger.info("Initializing MarketingBot")
        self.image_generator = ImageGenerator(openai_key)
        self.waapi_client = WaAPIClient(waapi_token, waapi_instance_id)
        logger.info("MarketingBot initialized with OpenAI and WaAPI")
    
    def process_request(self, user_id: str, product_image_path: str, product_details: Dict, product_type: str = "beverage") -> Dict:
        """Process a marketing image request"""
        try:
            logger.info(f"Processing request for user {user_id}")
            logger.info(f"Product details: {json.dumps(product_details)}")
            logger.info(f"Product image path: {product_image_path}")
            
            # Generate marketing image
            logger.info("Starting image generation")
            output_path = self.image_generator.generate_marketing_image(
                product_image_path,
                product_details,
                product_type
            )
            
            if output_path:
                logger.info(f"Image generated successfully: {output_path}")
                # Get the Railway app URL for serving images
                app_url = os.getenv('RAILWAY_STATIC_URL', os.getenv('APP_URL', 'https://auto-design-production.up.railway.app'))
                image_url = f"{app_url}/images/output/{os.path.basename(output_path)}"
                logger.info(f"Image URL: {image_url}")
                
                return {
                    "success": True,
                    "image_path": output_path,
                    "image_url": image_url
                }
            else:
                logger.error("Failed to generate marketing image")
                return {
                    "success": False,
                    "error": "Failed to generate marketing image"
                }
                
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

# Initialize marketing bot
logger.info("Initializing marketing bot with environment variables")
marketing_bot = MarketingBot(
    openai_key=OPENAI_API_KEY,
    waapi_token=WAAPI_API_TOKEN,
    waapi_instance_id=WAAPI_INSTANCE_ID
)

###################
# MESSAGE HANDLERS
###################

def handle_text_message(from_number: str, text: str):
    """Handle incoming text messages"""
    try:
        # Sanitize inputs for security
        from_number = str(from_number).strip()
        if text is None:
            text = ""
        text = str(text).strip()
        
        logger.info(f"Processing text message: '{text}' from {from_number}")
        
        # Create session if not exists
        if from_number not in user_sessions:
            logger.info(f"Creating new session for {from_number}")
            user_sessions[from_number] = {
                "product_image": None,
                "details": {},
                "state": "waiting_for_command",
                "last_activity": datetime.now().timestamp(),
                "created_at": datetime.now().timestamp()
            }
        else:
            # Update last activity timestamp
            user_sessions[from_number]["last_activity"] = datetime.now().timestamp()
        
        session = user_sessions[from_number]
        logger.info(f"Current session state: {session['state']}")
        
        # Check for start command (case insensitive)
        if text.lower() == 'edit':
            logger.info(f"User {from_number} sent 'edit' command")
            session['state'] = 'waiting_for_image'
            session['product_image'] = None
            session['details'] = {}
            logger.info(f"Session state changed to 'waiting_for_image'")
            
            marketing_bot.waapi_client.send_message(
                from_number,
                "Welcome to Marketing Image Editor! 📸\n\n"
                "Please send your product image to begin.\n\n"
                "After sending the image, I'll ask for details like company name, product name, price, etc."
            )
            logger.info(f"Sent welcome message to {from_number}")
            return
        
        # Check for generate command (case insensitive)
        if text.lower() == 'generate':
            logger.info(f"User {from_number} sent 'generate' command")
            # Validate we have all required info
            if not session.get('product_image'):
                logger.warning(f"User {from_number} tried to generate without an image")
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
                
                logger.warning(f"User {from_number} tried to generate with missing details: {missing}")
                marketing_bot.waapi_client.send_message(
                    from_number,
                    f"Missing required details: {', '.join(missing)}\n\n"
                    "Please provide all required information."
                )
                return
            
            # All required information is present, generate the image
            logger.info(f"Sending generation message to {from_number}")
            marketing_bot.waapi_client.send_message(
                from_number,
                "✨ Generating your marketing image...\n"
                "This may take a moment."
            )
            
            # Process in the background
            logger.info(f"Starting image generation process for {from_number}")
            result = marketing_bot.process_request(
                from_number,
                session['product_image'],
                details
            )
            
            if result['success']:
                logger.info(f"Image generated successfully for {from_number}: {result['image_path']}")
                # Get local file path
                image_path = result['image_path']
                
                # Read the image for base64 encoding
                logger.info(f"Reading image file for base64 encoding: {image_path}")
                with open(image_path, 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Send the generated image
                logger.info(f"Sending generated image to {from_number}")
                media_result = marketing_bot.waapi_client.send_media(
                    from_number,
                    "🎉 Here's your marketing image!\n\n"
                    "To create another image, send 'edit' again.",
                    media_base64=img_base64,
                    filename=os.path.basename(image_path)
                )
                
                if media_result['success']:
                    logger.info(f"Image sent successfully to {from_number}")
                else:
                    logger.error(f"Failed to send image: {media_result.get('error')}")
                    # Try to send an error message
                    marketing_bot.waapi_client.send_message(
                        from_number,
                        "I created your marketing image but couldn't send it. Please try again by sending 'edit'."
                    )
                
                # Reset state
                session['state'] = 'waiting_for_command'
                logger.info(f"Session state reset to 'waiting_for_command'")
            else:
                logger.error(f"Failed to generate image: {result.get('error')}")
                marketing_bot.waapi_client.send_message(
                    from_number,
                    f"Sorry, I couldn't generate the image: {result.get('error', 'Unknown error')}\n\n"
                    "Please try again or send 'edit' to start over."
                )
            return
        
        # Handle conversation flow based on state
        if session['state'] == 'waiting_for_image':
            logger.info(f"User {from_number} sent text while waiting for image")
            marketing_bot.waapi_client.send_message(
                from_number,
                "Please send your product image first.\n"
                "I'm waiting for an image file."
            )
            return
        
        elif session['state'] == 'waiting_for_details':
            logger.info(f"User {from_number} sent details: {text}")
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
                        logger.info(f"Set company_name: {value}")
                    elif 'product' in key:
                        session['details']['product_name'] = value
                        detail_provided = True
                        logger.info(f"Set product_name: {value}")
                    elif 'price' in key:
                        session['details']['price'] = value
                        detail_provided = True
                        logger.info(f"Set price: {value}")
                    elif 'tagline' in key:
                        session['details']['tagline'] = value
                        detail_provided = True
                        logger.info(f"Set tagline: {value}")
                    elif 'address' in key or 'location' in key:
                        session['details']['address'] = value
                        detail_provided = True
                        logger.info(f"Set address: {value}")
            
            # If no structured details, try to interpret single line inputs
            if not detail_provided:
                logger.info("No structured details found, interpreting as single value")
                if not session['details'].get('company_name'):
                    session['details']['company_name'] = text
                    logger.info(f"Set company_name: {text}")
                    detail_provided = True
                elif not session['details'].get('product_name'):
                    session['details']['product_name'] = text
                    logger.info(f"Set product_name: {text}")
                    detail_provided = True
                elif not session['details'].get('price'):
                    session['details']['price'] = text
                    logger.info(f"Set price: {text}")
                    detail_provided = True
                elif not session['details'].get('tagline'):
                    session['details']['tagline'] = text
                    logger.info(f"Set tagline: {text}")
                    detail_provided = True
                elif not session['details'].get('address'):
                    session['details']['address'] = text
                    logger.info(f"Set address: {text}")
                    detail_provided = True
            
            # Only send status update if user actually provided new details
            if detail_provided:
                logger.info(f"Sending status update to {from_number}")
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
            logger.info(f"User {from_number} sent message in default state")
            marketing_bot.waapi_client.send_message(
                from_number,
                "👋 Welcome to Marketing Image Generator!\n\n"
                "To create a marketing image, send 'edit' to start."
            )
            
    except Exception as e:
        logger.error(f"Error handling text message: {str(e)}", exc_info=True)
        try:
            marketing_bot.waapi_client.send_message(
                from_number,
                "Sorry, an error occurred. Please try again.\n"
                "Send 'edit' to start over."
            )
        except Exception as send_error:
            logger.error(f"Failed to send error message: {str(send_error)}")

def handle_image_message(from_number: str, media_data):
    """Handle incoming image messages"""
    try:
        logger.info(f"Image received from {from_number}. Processing...")
        
        # Create session if not exists
        if from_number not in user_sessions:
            logger.info(f"Creating new session for {from_number}")
            user_sessions[from_number] = {
                "product_image": None,
                "details": {},
                "state": "waiting_for_command"
            }
        
        session = user_sessions[from_number]
        
        # Debug session state
        logger.info(f"Current session state: {session['state']}")
        
        # Check if we're in the right state to receive an image
        if session['state'] != 'waiting_for_image':
            logger.warning(f"Received image but session state is {session['state']}, not waiting_for_image")
            marketing_bot.waapi_client.send_message(
                from_number,
                "I wasn't expecting an image right now.\n"
                "To start the process, please send 'edit' first."
            )
            return
        
        # Process the image from media_data (base64 encoded)
        try:
            if media_data and media_data.get('data'):  
                # Normal path - we have base64 data
                logger.info("Image data found. Decoding base64 data...")
                image_bytes = base64.b64decode(media_data['data'])
            else:
                # No media data in webhook - this is common with WaAPI's trial limitations
                # Create a placeholder image for testing
                logger.warning("No image data in webhook - creating placeholder image")
                
                # Save a placeholder image (1x1 transparent pixel)
                image_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
                
                # Let the user know about the limitation
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "⚠️ I received your image but couldn't access its data due to trial account limitations.\n\n"
                    "I'll continue with a placeholder image for testing purposes."
                )
            
            # Save the image
            filename = f"whatsapp_image_{from_number.replace('@c.us', '')}_{int(datetime.now().timestamp())}.jpg"
            image_path = os.path.join("images/input", filename)
            
            logger.info(f"Saving image to {image_path}")
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            logger.info(f"Image saved to {image_path}")
            session['product_image'] = image_path
            session['state'] = 'waiting_for_details'
            logger.info("Session state changed to 'waiting_for_details'")
            
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
            logger.info("Successfully processed image and sent response")
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            marketing_bot.waapi_client.send_message(
                from_number,
                "Sorry, I couldn't process your image. Please try again."
            )
            
    except Exception as e:
        logger.error(f"Error handling image message: {str(e)}", exc_info=True)
        try:
            marketing_bot.waapi_client.send_message(
                from_number,
                "Sorry, I couldn't process your image. Please try again.\n"
                "Start over by sending 'edit'."
            )
        except Exception as send_error:
            logger.error(f"Failed to send error message: {str(send_error)}")

###################
# FLASK ROUTES
###################

@app.route('/')
def home():
    """Render the home page"""
    logger.info("Home page accessed")
    return jsonify({
        "service": "Marketing Image Generator (WhatsApp Version)",
        "status": "running",
        "version": "1.2",
        "endpoint": "/webhook"
    })

@app.route('/health')
def health():
    """Health check endpoint for Railway"""
    try:
        # Check components health
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.2.0",
            "components": {
                "api": "healthy",
                "openai": "unknown",
                "waapi": "unknown"
            }
        }
        
        # Check OpenAI connection
        try:
            # Basic API validation without actually making a costly request
            if OPENAI_API_KEY and len(OPENAI_API_KEY) > 20:
                health_status["components"]["openai"] = "healthy"
            else:
                health_status["components"]["openai"] = "degraded"
        except Exception:
            health_status["components"]["openai"] = "unhealthy"
            
        # Check WaAPI connection
        try:
            if WAAPI_API_TOKEN and WAAPI_INSTANCE_ID:
                health_status["components"]["waapi"] = "healthy"
            else:
                health_status["components"]["waapi"] = "degraded"
        except Exception:
            health_status["components"]["waapi"] = "unhealthy"
            
        # Set overall status based on components
        if "unhealthy" in health_status["components"].values():
            health_status["status"] = "unhealthy"
        elif "degraded" in health_status["components"].values():
            health_status["status"] = "degraded"
            
        logger.debug("Health check accessed")
        return jsonify(health_status)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/images/<path:path>')
def serve_images(path):
    """Serve images from the images directory"""
    logger.debug(f"Serving image: {path}")
    directory, filename = os.path.split(path)
    return send_from_directory(os.path.join('images', directory), filename)

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages via WaAPI webhook"""
    try:
        # Check request size for security
        content_length = request.content_length
        if content_length and content_length > 10 * 1024 * 1024:  # 10MB limit
            logger.warning(f"Request too large: {content_length} bytes")
            return jsonify({"status": "error", "message": "Request too large"}), 413
        
        # Verify webhook secret if configured
        if WEBHOOK_SECRET:
            signature = request.headers.get('X-Webhook-Signature')
            if not signature:
                logger.warning("Missing webhook signature")
                return jsonify({"status": "error", "message": "Missing signature"}), 401
            
            # Implement your signature verification here
            # This is a placeholder for actual verification
            # if not verify_signature(request.data, signature, WEBHOOK_SECRET):
            #     logger.warning("Invalid webhook signature")
            #     return jsonify({"status": "error", "message": "Invalid signature"}), 401
        
        # Extract the webhook data
        try:
            webhook_data = request.json
        except Exception as json_error:
            logger.error(f"Invalid JSON in webhook request: {str(json_error)}")
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400
            
        logger.info(f"Received webhook: {json.dumps(webhook_data)}")
        
        # Extract message ID to prevent duplicate processing
        message_data = webhook_data.get('data', {}).get('message', {})
        
        # Debug the message structure
        logger.debug(f"Message data structure: {json.dumps(message_data)}")
        
        message_id = message_data.get('id', {})
        if isinstance(message_id, dict):
            serialized_id = message_id.get('_serialized', '')
        else:
            serialized_id = str(message_id)
        
        # Skip if we've already processed this message
        if serialized_id and serialized_id in processed_messages:
            logger.info(f"Skipping duplicate message: {serialized_id}")
            return jsonify({"status": "success", "message": "Duplicate message skipped"})
        
        # Mark as processed if it has an ID
        if serialized_id:
            processed_messages[serialized_id] = datetime.now().timestamp()
            logger.debug(f"Added message {serialized_id} to processed cache")
            
            # Limit cache size by removing old entries (keep last 100)
            if len(processed_messages) > 100:
                oldest = sorted(processed_messages.items(), key=lambda x: x[1])[0][0]
                processed_messages.pop(oldest)
                logger.debug(f"Removed oldest message {oldest} from cache")
        
        # Check if this is a message event
        if webhook_data.get('event') in ['message', 'message_create']:
            # Extract message details - safer parsing with fallbacks
            message_type = message_data.get('type', '')
            from_number = message_data.get('from', '')
            # Handle None values properly when parsing body
            raw_body = message_data.get('body')
            body = raw_body.strip().lower() if isinstance(raw_body, str) else ''
            
            # Check for media in multiple ways
            has_media = (
                message_data.get('hasMedia', False) or 
                message_type in ['image', 'sticker', 'video', 'document'] or
                'mediaData' in message_data or
                '_data' in message_data and message_data['_data'].get('type') in ['image', 'sticker', 'video', 'document']
            )
            
            logger.info(f"Received message from {from_number}: {body}, Media: {has_media}, Type: {message_type}")
            
            # Check if the number is in the proper format
            if '@c.us' not in str(from_number):
                logger.error(f"Invalid phone number format: {from_number}")
                return jsonify({"status": "error", "message": "Invalid phone number format"})
            
            # Try to get media data in multiple ways
            media_data = webhook_data.get('data', {}).get('media', {})
            if not media_data:
                logger.debug("No direct media data found, checking alternative locations")
                if '_data' in message_data and message_data['_data'].get('type') in ['image', 'sticker', 'video', 'document']:
                    logger.debug("Found media in _data")
                    # Try to extract media from message_data itself for different webhook formats
                    if hasattr(message_data, 'mediaData') or 'mediaData' in message_data:
                        media_data = message_data.get('mediaData', {})
                    elif hasattr(message_data, 'mimetype') or 'mimetype' in message_data:
                        # It might be directly in the message
                        media_data = message_data
            
            # Handle media messages
            if has_media:
                logger.info(f"Detected media in message from {from_number}, type: {message_type}")
                if not media_data:
                    # If webhook doesn't include media data, try to fetch it
                    logger.info("Media indicated but no media data in webhook, will attempt download")
                
                # Process as image regardless of empty media data - handler will check session state
                handle_image_message(from_number, media_data)
                return jsonify({"status": "success", "message": "Media processed"})
                
            # Handle text messages
            handle_text_message(from_number, body)
            return jsonify({"status": "success", "message": "Text processed"})
        
        # Handle other event types if needed
        logger.debug(f"Event {webhook_data.get('event')} processed")
        return jsonify({"status": "success", "message": "Event processed"})
    
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}", exc_info=True)  # Added full traceback
        return jsonify({"status": "error", "message": str(e)})

###################
# Session cleanup function
def cleanup_expired_sessions():
    """Remove expired user sessions to prevent memory leaks"""
    try:
        current_time = datetime.now().timestamp()
        expired_session_count = 0
        expired_keys = []
        
        # Find expired sessions
        for user_id, session in user_sessions.items():
            last_activity = session.get('last_activity', 0)
            if current_time - last_activity > SESSION_TIMEOUT_HOURS * 3600:
                expired_keys.append(user_id)
                expired_session_count += 1
        
        # Remove expired sessions
        for key in expired_keys:
            del user_sessions[key]
            
        if expired_session_count > 0:
            logger.info(f"Cleaned up {expired_session_count} expired sessions")
    except Exception as e:
        logger.error(f"Error cleaning up sessions: {str(e)}", exc_info=True)

# Add a scheduled task to clean up sessions
def schedule_cleanup():
    cleanup_expired_sessions()
    # Schedule next cleanup in 1 hour
    threading.Timer(3600, schedule_cleanup).start()

###################
# MAIN APPLICATION
###################

if __name__ == '__main__':
    import threading
    
    # Start the session cleanup scheduler
    schedule_cleanup()
    
    # Get port from environment variable (Railway provides this)
    port = int(os.getenv('PORT', 5000))
    
    # Production startup logging
    logger.info(f"Starting Marketing Bot API (WhatsApp Version) v1.2.0 on port {port}")
    logger.info(f"OpenAI client initialized: {OPENAI_VERSION}")
    logger.info(f"WaAPI instance ID: {WAAPI_INSTANCE_ID}")
    logger.info(f"Environment: {'Production' if not os.getenv('DEBUG') else 'Development'}")
    
    # Run the Flask app with production settings
    app.run(
        host='0.0.0.0', 
        port=port,
        debug=False,  # Always false in production
        threaded=True,
        use_reloader=False  # Disable reloader in production
    )