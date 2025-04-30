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
import time
import traceback
from datetime import datetime
from typing import Optional, Dict, Any
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MarketingBotWhatsApp")

# Also add print statements for important logs to ensure they appear in Railway logs
def log_and_print(level, message):
    if level == "INFO":
        logger.info(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "DEBUG":
        logger.debug(message)
    
    # Always print to stdout for Railway logs
    print(f"[{level}] {message}")

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Create directories for storing images
os.makedirs('images/input', exist_ok=True)
os.makedirs('images/output', exist_ok=True)

# Store user sessions
user_sessions = {}

# Store processed messages to prevent duplicates
processed_messages = {}

# Store last message time for rate limiting
last_message_time = {}

# Initialize OpenAI with API key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    log_and_print("ERROR", "OPENAI_API_KEY not found in environment variables!")
    raise ValueError("OPENAI_API_KEY is required")

# Initialize OpenAI client
from openai import OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)
log_and_print("INFO", "OpenAI client initialized with API key")
print(f"OpenAI API Key first 8 chars: {OPENAI_API_KEY[:8] if len(OPENAI_API_KEY) >= 8 else 'Too short'}...")

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
    log_and_print("ERROR", f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

log_and_print("INFO", f"Environment variables loaded. Instance ID: {WAAPI_INSTANCE_ID}")

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
        log_and_print("INFO", "Initializing ImageGenerator")
        self.api_key = api_key
        self.client = openai_client
        log_and_print("INFO", "ImageGenerator initialized with OpenAI API")
    
    def generate_marketing_image(self, product_image_path: str, product_details: Dict, product_type: str = "beverage") -> Optional[str]:
        """Generate a marketing image using OpenAI API"""
        try:
            product_name = product_details.get('product_name', 'product')
            log_and_print("INFO", f"Starting image generation for {product_name} as {product_type}")
            
            # Select the appropriate prompt template
            if product_type.lower() == "beverage":
                prompt = PromptTemplates.get_beverage_template(product_details)
                log_and_print("INFO", "Using beverage template")
            elif product_type.lower() == "food":
                prompt = PromptTemplates.get_food_template(product_details)
                log_and_print("INFO", "Using food template")
            else:
                prompt = PromptTemplates.get_master_template(product_details)
                log_and_print("INFO", "Using master template")
            
            # Generate the image
            try:
                log_and_print("INFO", "Sending image generation request to OpenAI API with gpt-image-1 model")
                print(f"[DEBUG] Full prompt: {prompt[:100]}...")  # Print first 100 chars
                print(f"[DEBUG] Product details: {json.dumps(product_details)}")
                print(f"[DEBUG] Product image path: {product_image_path}")
                print(f"[DEBUG] Image exists: {os.path.exists(product_image_path)}")
                
                # Add retries for production reliability
                max_retries = 3
                retry_delay = 2
                
                for retry in range(max_retries):
                    try:
                        print(f"[DEBUG] API call attempt {retry+1} to OpenAI")
                        
                        # Make API call
                        result = self.client.images.generate(
                            model="gpt-image-1",
                            prompt=prompt,
                            size="1024x1024",
                            n=1,
                            response_format="b64_json",
                            quality="low"  # Use low quality for production to save credits
                        )
                        print(f"[DEBUG] API call successful on attempt {retry+1}")
                        # If successful, break the retry loop
                        break
                    except Exception as retry_error:
                        error_message = str(retry_error)
                        log_and_print("WARNING", f"API call attempt {retry+1} failed: {error_message}")
                        
                        # Print detailed error information
                        print(f"[ERROR] OpenAI API error details: {error_message}")
                        print(f"[ERROR] Error type: {type(retry_error).__name__}")
                        
                        if "rate limit" in error_message.lower():
                            print("[ERROR] Rate limit exceeded. Waiting longer before retry.")
                            retry_delay = 10  # Wait longer for rate limit errors
                        
                        if retry < max_retries - 1:
                            log_and_print("INFO", f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            # Last attempt failed, re-raise the exception
                            log_and_print("ERROR", "All API call retries failed")
                            raise
                
                log_and_print("INFO", "OpenAI API request completed")
                
                # Print the result structure for debugging
                print(f"[DEBUG] OpenAI result type: {type(result)}")
                print(f"[DEBUG] OpenAI result has data attribute: {hasattr(result, 'data')}")
                if hasattr(result, 'data'):
                    print(f"[DEBUG] result.data length: {len(result.data)}")
                    if len(result.data) > 0:
                        print(f"[DEBUG] First item type: {type(result.data[0])}")
                        print(f"[DEBUG] First item has b64_json: {hasattr(result.data[0], 'b64_json')}")
                
                if hasattr(result, 'data') and len(result.data) > 0 and hasattr(result.data[0], 'b64_json'):
                    log_and_print("INFO", "Image base64 data received, decoding")
                    image_bytes = base64.b64decode(result.data[0].b64_json)
                    log_and_print("INFO", "Image decoded successfully")
                    print(f"[DEBUG] Decoded image bytes length: {len(image_bytes)}")
                else:
                    log_and_print("ERROR", "No image data in OpenAI response")
                    print("[ERROR] OpenAI response structure differs from expected format")
                    return {"error": "Failed to get image data from OpenAI response"}
                    
                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                product_name_safe = product_details.get('product_name', 'product').replace(' ', '_')[:20]
                output_filename = f"{product_name_safe}_{timestamp}.png"
                output_path = os.path.join("images/output", output_filename)
                print(f"[DEBUG] Image will be saved to: {output_path}")
                
                # Verify output directory exists
                os.makedirs("images/output", exist_ok=True)
                print(f"[DEBUG] Verified output directory exists")
                
                # Convert to image
                log_and_print("INFO", "Creating image from bytes")
                try:
                    image = Image.open(BytesIO(image_bytes))
                    print(f"[DEBUG] Image opened successfully: {image.format} {image.size}px")
                except Exception as img_error:
                    print(f"[ERROR] Failed to open image: {str(img_error)}")
                    return {"error": f"Failed to open generated image: {str(img_error)}"}
                
                # Optionally resize for optimization
                if image.size[0] > 1500 or image.size[1] > 1500:
                    log_and_print("INFO", f"Resizing image from {image.size} to max 1500px")
                    image.thumbnail((1500, 1500), Image.LANCZOS)
                    print(f"[DEBUG] Image resized to: {image.size}px")
                
                # Save the image
                log_and_print("INFO", f"Saving image to {output_path}")
                try:
                    image.save(output_path, format="PNG", optimize=True)
                    print(f"[DEBUG] Image saved successfully as PNG")
                except Exception as save_error:
                    print(f"[ERROR] Failed to save image: {str(save_error)}")
                    return {"error": f"Failed to save image: {str(save_error)}"}
                
                log_and_print("INFO", f"Marketing image saved to {output_path}")
                return output_path
                
            except Exception as api_error:
                error_message = str(api_error)
                log_and_print("ERROR", f"OpenAI API Error: {error_message}")
                print(f"[ERROR] Failed to generate image with OpenAI API: {error_message}")
                print(f"[ERROR] Error type: {type(api_error).__name__}")
                
                # Return a clear error message instead of None
                return {"error": f"OpenAI API error: {error_message}"}
                
        except Exception as e:
            error_message = str(e)
            log_and_print("ERROR", f"Error generating marketing image: {error_message}")
            print(f"[ERROR] Exception in generate_marketing_image: {error_message}")
            traceback.print_exc()  # Print full stack trace for debugging
            return {"error": f"Error in image generation process: {error_message}"}

###################
# WAAPI WHATSAPP API
###################

class WaAPIClient:
    def __init__(self, api_token: str, instance_id: str):
        log_and_print("INFO", f"Initializing WaAPIClient for instance {instance_id}")
        self.api_token = api_token
        self.instance_id = instance_id
        self.api_base_url = "https://waapi.app/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        log_and_print("INFO", "WaAPIClient initialized successfully")
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make a request to the WaAPI API"""
        url = f"{self.api_base_url}/{endpoint}"
        log_and_print("INFO", f"Making {method} request to {endpoint}")
        try:
            if method.lower() == "get":
                response = requests.get(url, headers=self.headers)
            elif method.lower() == "post":
                print(f"[DEBUG] POST data: {json.dumps(data)[:200]}..." if data else "None")
                response = requests.post(url, headers=self.headers, json=data)
            elif method.lower() == "put":
                response = requests.put(url, headers=self.headers, json=data)
            elif method.lower() == "delete":
                response = requests.delete(url, headers=self.headers)
            else:
                log_and_print("ERROR", f"Invalid method: {method}")
                return {"success": False, "error": "Invalid method"}
            
            response.raise_for_status()
            result = response.json()
            print(f"[DEBUG] Response: {json.dumps(result)[:200]}...")
            return result
        except Exception as e:
            log_and_print("ERROR", f"Error making request to WaAPI: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_instance_status(self) -> Dict:
        """Get the status of the instance"""
        log_and_print("INFO", "Checking instance status")
        return self._make_request("get", f"instances/{self.instance_id}/client/status")
    
    def send_message(self, to: str, message: str) -> Dict:
        """Send a text message via WhatsApp"""
        # Format phone number to WhatsApp format if needed
        to = self._format_phone_number(to)
        
        # Apply rate limiting
        current_time = time.time()
        if to in last_message_time and current_time - last_message_time[to] < 2:
            log_and_print("INFO", f"Rate limiting message to {to} - too soon after last message")
            time.sleep(2 - (current_time - last_message_time[to]))
            current_time = time.time()
        
        data = {
            "chatId": to,
            "message": message
        }
        
        log_and_print("INFO", f"Sending message to {to}: {message[:50]}{'...' if len(message) > 50 else ''}")
        result = self._make_request(
            "post", 
            f"instances/{self.instance_id}/client/action/send-message", 
            data
        )
        
        if result.get("status") == "success":
            log_and_print("INFO", f"Message sent successfully to {to}")
            # Update rate limiting
            last_message_time[to] = current_time
            return {"success": True, "data": result.get("data", {})}
        else:
            log_and_print("ERROR", f"Error sending message: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get("error", "Unknown error")}
    
    def send_media(self, to: str, caption: str = "", media_url: str = None, media_base64: str = None, 
                  filename: str = None, is_sticker: bool = False) -> Dict:
        """Send a media message via WhatsApp"""
        # Format phone number to WhatsApp format if needed
        to = self._format_phone_number(to)
        
        # Apply rate limiting
        current_time = time.time()
        if to in last_message_time and current_time - last_message_time[to] < 2:
            log_and_print("INFO", f"Rate limiting media message to {to} - too soon after last message")
            time.sleep(2 - (current_time - last_message_time[to]))
            current_time = time.time()
        
        data = {
            "chatId": to,
            "mediaCaption": caption
        }
        
        # Add either URL or base64 data
        if media_url:
            log_and_print("INFO", f"Sending media from URL to {to}")
            data["mediaUrl"] = media_url
        elif media_base64:
            log_and_print("INFO", f"Sending media from base64 to {to}")
            data["mediaBase64"] = media_base64
            # Filename is required for base64 media
            data["mediaName"] = filename or f"media_{int(datetime.now().timestamp())}.png"
        else:
            log_and_print("ERROR", "No media URL or base64 data provided")
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
            log_and_print("INFO", f"Media sent successfully to {to}")
            # Update rate limiting
            last_message_time[to] = current_time
            return {"success": True, "data": result.get("data", {})}
        else:
            log_and_print("ERROR", f"Error sending media: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get("error", "Unknown error")}
    
    def _format_phone_number(self, phone_number: str) -> str:
        """Format phone number to the expected format for WhatsApp"""
        print(f"[DEBUG] Formatting phone number: {phone_number}")
        # Check if it's already in a valid format (either individual or group chat)
        if '@c.us' in phone_number or '@g.us' in phone_number:
            return phone_number
            
        # Strip any non-numeric characters (like +, spaces, etc.)
        clean_number = ''.join(filter(str.isdigit, phone_number))
        
        # Format for WhatsApp: number@c.us
        result = f"{clean_number}@c.us"
        print(f"[DEBUG] Formatted phone number: {result}")
        return result

###################
# MARKETING BOT
###################

class MarketingBot:
    def __init__(self, openai_key: str, waapi_token: str, waapi_instance_id: str):
        log_and_print("INFO", "Initializing MarketingBot")
        self.image_generator = ImageGenerator(openai_key)
        self.waapi_client = WaAPIClient(waapi_token, waapi_instance_id)
        log_and_print("INFO", "MarketingBot initialized with OpenAI and WaAPI")
    
    def process_request(self, user_id: str, product_image_path: str, product_details: Dict, product_type: str = "beverage") -> Dict:
        """Process a marketing image request"""
        try:
            log_and_print("INFO", f"Processing request for user {user_id}")
            print(f"[DEBUG] Product details: {json.dumps(product_details)}")
            print(f"[DEBUG] Product image path: {product_image_path}")
            print(f"[DEBUG] File exists check: {os.path.exists(product_image_path)}")
            
            # Check if product image exists
            if not os.path.exists(product_image_path):
                error_msg = f"Product image not found at {product_image_path}"
                log_and_print("ERROR", error_msg)
                return {"success": False, "error": error_msg}
            
            # Generate marketing image
            log_and_print("INFO", "Starting image generation")
            result = self.image_generator.generate_marketing_image(
                product_image_path,
                product_details,
                product_type
            )
            
            # Check if result is an error message (dict with error key)
            if isinstance(result, dict) and "error" in result:
                log_and_print("ERROR", f"Image generation failed: {result['error']}")
                return {"success": False, "error": result["error"]}
            
            # Normal success case
            if result:
                log_and_print("INFO", f"Image generated successfully: {result}")
                # Get the Railway app URL for serving images
                app_url = os.getenv('RAILWAY_STATIC_URL', os.getenv('APP_URL', 'https://auto-design-production.up.railway.app'))
                image_url = f"{app_url}/images/output/{os.path.basename(result)}"
                log_and_print("INFO", f"Image URL: {image_url}")
                
                return {
                    "success": True,
                    "image_path": result,
                    "image_url": image_url
                }
            else:
                error_msg = "Failed to generate marketing image"
                log_and_print("ERROR", error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = str(e)
            log_and_print("ERROR", f"Error processing request: {error_msg}")
            print(f"[ERROR] Exception type: {type(e).__name__}")
            print(f"[ERROR] Exception details: {error_msg}")
            traceback.print_exc()  # Print full stack trace for debugging
            return {
                "success": False,
                "error": error_msg
            }

# Initialize marketing bot
log_and_print("INFO", "Initializing marketing bot with environment variables")
marketing_bot = MarketingBot(
    openai_key=OPENAI_API_KEY,
    waapi_token=WAAPI_API_TOKEN,
    waapi_instance_id=WAAPI_INSTANCE_ID
)

###################
# MESSAGE HANDLERS
###################

def handle_text_message(from_number: str, text: str, participant: str = None):
    """Handle incoming text messages"""
    try:
        # For group chats, the actual sender is in the participant field
        actual_sender = participant if participant else from_number
        
        print(f"[DEBUG] Received text: '{text}' from {from_number}")
        if participant:
            print(f"[DEBUG] Message sent by participant: {participant} in group: {from_number}")
        
        log_and_print("INFO", f"Processing text message: '{text}' from {actual_sender}")
        
        # For direct messages, we'll handle any text
        # For group messages, we'll only handle commands
        is_group_chat = '@g.us' in from_number
        is_command = text.lower() in ['edit', 'generate'] or text.lower().startswith('/edit') or text.lower().startswith('/generate')
        
        print(f"[DEBUG] Is group chat: {is_group_chat}")
        print(f"[DEBUG] Is command: {is_command}")
        print(f"[DEBUG] Text lower: '{text.lower()}'")
        
        # Only process direct messages or commands in groups
        if is_group_chat and not is_command:
            print(f"[DEBUG] Ignoring non-command group message")
            return
        
        # Remove command prefix if present (for group chat commands)
        cleaned_text = text.lower()
        if cleaned_text.startswith('/edit'):
            cleaned_text = 'edit'
        elif cleaned_text.startswith('/generate'):
            cleaned_text = 'generate'
        
        print(f"[DEBUG] Cleaned text: '{cleaned_text}'")
        
        # Create session if not exists
        if actual_sender not in user_sessions:
            log_and_print("INFO", f"Creating new session for {actual_sender}")
            user_sessions[actual_sender] = {
                "product_image": None,
                "details": {},
                "state": "waiting_for_command"
            }
            print(f"[DEBUG] New session created with state 'waiting_for_command'")
        
        session = user_sessions[actual_sender]
        log_and_print("INFO", f"Current session state: {session['state']}")
        
        # Check for start command
        if cleaned_text == 'edit':
            log_and_print("INFO", f"User {actual_sender} sent 'edit' command")
            session['state'] = 'waiting_for_image'
            session['product_image'] = None
            session['details'] = {}
            log_and_print("INFO", f"Session state changed to 'waiting_for_image'")
            
            response = marketing_bot.waapi_client.send_message(
                actual_sender,
                "Welcome to Marketing Image Editor! 📸\n\n"
                "Please send your product image to begin.\n\n"
                "After sending the image, I'll ask for details like company name, product name, price, etc."
            )
            
            print(f"[DEBUG] Send message response: {json.dumps(response)}")
            log_and_print("INFO", f"Sent welcome message to {actual_sender}")
            return
        
        # Check for generate command
        if cleaned_text == 'generate':
            print(f"[DEBUG] 'generate' command detected from {actual_sender}")
            log_and_print("INFO", f"User {actual_sender} sent 'generate' command")
            
            # Print current session state for debugging
            print(f"[DEBUG] Current session state: {session['state']}")
            print(f"[DEBUG] Product image path: {session.get('product_image')}")
            print(f"[DEBUG] Details collected: {json.dumps(session.get('details', {}))}")
            
            # Validate we have all required info
            if not session.get('product_image'):
                log_and_print("WARNING", f"User {actual_sender} tried to generate without an image")
                print(f"[DEBUG] No product image in session")
                marketing_bot.waapi_client.send_message(
                    actual_sender,
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
                
                log_and_print("WARNING", f"User {actual_sender} tried to generate with missing details: {missing}")
                marketing_bot.waapi_client.send_message(
                    actual_sender,
                    f"Missing required details: {', '.join(missing)}\n\n"
                    "Please provide all required information."
                )
                return
            
            # Generate the image
            log_and_print("INFO", f"Sending generation message to {actual_sender}")
            marketing_bot.waapi_client.send_message(
                actual_sender,
                "✨ Generating your marketing image...\n"
                "This may take up to 30 seconds."
            )
            
            # Process in the background
            log_and_print("INFO", f"Starting image generation process for {actual_sender}")
            result = marketing_bot.process_request(
                actual_sender,
                session['product_image'],
                details
            )
            
            if result['success']:
                log_and_print("INFO", f"Image generated successfully for {actual_sender}: {result['image_path']}")
                # Get local file path
                image_path = result['image_path']
                
                # Check if file exists
                if not os.path.exists(image_path):
                    log_and_print("ERROR", f"Generated image file not found: {image_path}")
                    marketing_bot.waapi_client.send_message(
                        actual_sender,
                        "Sorry, there was a problem saving the generated image. Please try again."
                    )
                    # Reset state
                    session['state'] = 'waiting_for_command'
                    return
                
                # Read the image for base64 encoding
                log_and_print("INFO", f"Reading image file for base64 encoding: {image_path}")
                print(f"[DEBUG] Checking if file exists: {os.path.exists(image_path)}")
                print(f"[DEBUG] File size: {os.path.getsize(image_path) if os.path.exists(image_path) else 'not found'} bytes")
                
                try:
                    with open(image_path, 'rb') as img_file:
                        img_data = img_file.read()
                        print(f"[DEBUG] Image data read: {len(img_data)} bytes")
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        print(f"[DEBUG] Base64 encoding length: {len(img_base64)} characters")
                except Exception as read_error:
                    log_and_print("ERROR", f"Failed to read image file: {str(read_error)}")
                    marketing_bot.waapi_client.send_message(
                        actual_sender,
                        "Sorry, I had trouble processing the generated image. Please try again."
                    )
                    # Reset state to allow retry
                    session['state'] = 'waiting_for_command'
                    return
                
                # Send the generated image
                log_and_print("INFO", f"Sending generated image to {actual_sender}")
                print(f"[DEBUG] Sending image through WaAPI with caption")
                media_result = marketing_bot.waapi_client.send_media(
                    actual_sender,
                    "🎉 Here's your marketing image!\n\n"
                    "To create another image, send 'edit' again.",
                    media_base64=img_base64,
                    filename=os.path.basename(image_path)
                )
                print(f"[DEBUG] WaAPI media send result: {json.dumps(media_result)}")
                
                if media_result['success']:
                    log_and_print("INFO", f"Image sent successfully to {actual_sender}")
                else:
                    log_and_print("ERROR", f"Failed to send image: {media_result.get('error')}")
                    # Try to send an error message
                    marketing_bot.waapi_client.send_message(
                        actual_sender,
                        "I created your marketing image but couldn't send it. Please try again by sending 'edit'."
                    )
                
                # Reset state
                session['state'] = 'waiting_for_command'
                log_and_print("INFO", f"Session state reset to 'waiting_for_command'")
            else:
                log_and_print("ERROR", f"Failed to generate image: {result.get('error')}")
                marketing_bot.waapi_client.send_message(
                    actual_sender,
                    f"Sorry, I couldn't generate the image. {result.get('error', '')}\n\n"
                    "Please try again by sending 'edit'."
                )
                # Reset state to allow retry
                session['state'] = 'waiting_for_command'
            return
        
        # Handle conversation flow based on state
        if session['state'] == 'waiting_for_image':
            log_and_print("INFO", f"User {actual_sender} sent text while waiting for image")
            marketing_bot.waapi_client.send_message(
                actual_sender,
                "Please send your product image first.\n"
                "I'm waiting for an image file."
            )
            return
        
        elif session['state'] == 'waiting_for_details':
            log_and_print("INFO", f"User {actual_sender} sent details: {cleaned_text}")
            # Parse the details
            lines = text.split('\n')  # Use original text for details, not lowercase
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
                        log_and_print("INFO", f"Set company_name: {value}")
                    elif 'product' in key:
                        session['details']['product_name'] = value
                        detail_provided = True
                        log_and_print("INFO", f"Set product_name: {value}")
                    elif 'price' in key:
                        session['details']['price'] = value
                        detail_provided = True
                        log_and_print("INFO", f"Set price: {value}")
                    elif 'tagline' in key:
                        session['details']['tagline'] = value
                        detail_provided = True
                        log_and_print("INFO", f"Set tagline: {value}")
                    elif 'address' in key or 'location' in key:
                        session['details']['address'] = value
                        detail_provided = True
                        log_and_print("INFO", f"Set address: {value}")
            
            # If no structured details, try to interpret single line inputs
            if not detail_provided:
                log_and_print("INFO", "No structured details found, interpreting as single value")
                if not session['details'].get('company_name'):
                    session['details']['company_name'] = text  # Use original text
                    log_and_print("INFO", f"Set company_name: {text}")
                    detail_provided = True
                elif not session['details'].get('product_name'):
                    session['details']['product_name'] = text  # Use original text
                    log_and_print("INFO", f"Set product_name: {text}")
                    detail_provided = True
                elif not session['details'].get('price'):
                    session['details']['price'] = text  # Use original text
                    log_and_print("INFO", f"Set price: {text}")
                    detail_provided = True
                elif not session['details'].get('tagline'):
                    session['details']['tagline'] = text  # Use original text
                    log_and_print("INFO", f"Set tagline: {text}")
                    detail_provided = True
                elif not session['details'].get('address'):
                    session['details']['address'] = text  # Use original text
                    log_and_print("INFO", f"Set address: {text}")
                    detail_provided = True
            
            # Send updated status and next step
            log_and_print("INFO", f"Sending status update to {actual_sender}")
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
            
            marketing_bot.waapi_client.send_message(actual_sender, status_msg)
            return
        
        # Default state - for direct messages
        else:
            # Always respond to direct messages, only respond to commands in groups
            if not is_group_chat or is_command:
                log_and_print("INFO", f"User {actual_sender} sent message in default state")
                marketing_bot.waapi_client.send_message(
                    actual_sender,
                    "👋 Welcome to Marketing Image Generator!\n\n"
                    "To create a marketing image, send 'edit' to start."
                )
            
    except Exception as e:
        log_and_print("ERROR", f"Error handling text message: {str(e)}")
        print(f"[ERROR] Exception in handle_text_message: {str(e)}")
        traceback.print_exc()  # Print the full stack trace
        try:
            marketing_bot.waapi_client.send_message(
                from_number if '@g.us' not in from_number else participant,
                "Sorry, an error occurred. Please try again.\n"
                "Send 'edit' to start over."
            )
        except Exception as send_error:
            log_and_print("ERROR", f"Failed to send error message: {str(send_error)}")

def handle_image_message(from_number: str, media_data, participant: str = None):
    """Handle incoming image messages"""
    try:
        # For group chats, the actual sender is in the participant field
        actual_sender = participant if participant else from_number
        
        log_and_print("INFO", f"Image received from {actual_sender}. Processing...")
        if participant:
            print(f"[DEBUG] Image sent by participant: {participant} in group: {from_number}")
            
        print(f"[DEBUG] Media data present: {'Yes' if media_data else 'No'}")
        if media_data:
            print(f"[DEBUG] Media data keys: {list(media_data.keys()) if isinstance(media_data, dict) else 'Not a dict'}")
        
        # Create session if not exists
        if actual_sender not in user_sessions:
            log_and_print("INFO", f"Creating new session for {actual_sender}")
            user_sessions[actual_sender] = {
                "product_image": None,
                "details": {},
                "state": "waiting_for_command"
            }
        
        session = user_sessions[actual_sender]
        
        # Debug session state
        log_and_print("INFO", f"Current session state: {session['state']}")
        
        # Check if we're in the right state to receive an image
        if session['state'] != 'waiting_for_image':
            log_and_print("WARNING", f"Received image but session state is {session['state']}, not waiting_for_image")
            marketing_bot.waapi_client.send_message(
                actual_sender,
                "I wasn't expecting an image right now.\n"
                "To start the process, please send 'edit' first."
            )
            return
        
        # Process the image from media_data (base64 encoded)
        try:
            if media_data and media_data.get('data'):  
                # Normal path - we have base64 data
                log_and_print("INFO", "Image data found. Decoding base64 data...")
                print(f"[DEBUG] Base64 data length: {len(media_data['data'])}")
                image_bytes = base64.b64decode(media_data['data'])
                print(f"[DEBUG] Decoded image bytes length: {len(image_bytes)}")
            else:
                # No media data in webhook - this is common with WaAPI's trial limitations
                log_and_print("WARNING", "No image data in webhook - creating placeholder image")
                print("[DEBUG] Creating a 1x1 placeholder image")
                
                # Save a placeholder image (1x1 transparent pixel)
                image_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
                
                # Let the user know about the limitation
                marketing_bot.waapi_client.send_message(
                    actual_sender,
                    "⚠️ I received your image but couldn't access its data due to trial account limitations.\n\n"
                    "I'll continue with a placeholder image for testing purposes."
                )
            
            # Save the image
            filename = f"whatsapp_image_{actual_sender.replace('@c.us', '').replace('@g.us', '')}_{int(datetime.now().timestamp())}.jpg"
            image_path = os.path.join("images/input", filename)
            
            log_and_print("INFO", f"Saving image to {image_path}")
            try:
                # Make sure the directory exists
                os.makedirs("images/input", exist_ok=True)
                
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
                print(f"[DEBUG] Image saved successfully to {image_path}")
                print(f"[DEBUG] File exists: {os.path.exists(image_path)}")
                print(f"[DEBUG] File size: {os.path.getsize(image_path)} bytes")
            except Exception as save_error:
                log_and_print("ERROR", f"Failed to save image: {str(save_error)}")
                marketing_bot.waapi_client.send_message(
                    actual_sender,
                    "Sorry, I couldn't save your image. Please try again."
                )
                return
            
            log_and_print("INFO", f"Image saved to {image_path}")
            session['product_image'] = image_path
            session['state'] = 'waiting_for_details'
            log_and_print("INFO", "Session state changed to 'waiting_for_details'")
            
            marketing_bot.waapi_client.send_message(
                actual_sender,
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
            log_and_print("INFO", "Successfully processed image and sent response")
            
        except Exception as e:
            log_and_print("ERROR", f"Error processing image: {str(e)}")
            print(f"[ERROR] Exception in image processing: {str(e)}")
            traceback.print_exc()  # Print full stack trace
            marketing_bot.waapi_client.send_message(
                actual_sender,
                "Sorry, I couldn't process your image. Please try again."
            )
            
    except Exception as e:
        log_and_print("ERROR", f"Error handling image message: {str(e)}")
        print(f"[ERROR] Exception in handle_image_message: {str(e)}")
        traceback.print_exc()  # Print full stack trace
        try:
            marketing_bot.waapi_client.send_message(
                from_number if '@g.us' not in from_number else participant,
                "Sorry, I couldn't process your image. Please try again.\n"
                "Start over by sending 'edit'."
            )
        except Exception as send_error:
            log_and_print("ERROR", f"Failed to send error message: {str(send_error)}")

###################
# FLASK ROUTES
###################

@app.route('/')
def home():
    """Render the home page"""
    log_and_print("INFO", "Home page accessed")
    return jsonify({
        "service": "Marketing Image Generator (WhatsApp Version)",
        "status": "running",
        "version": "1.0",
        "endpoint": "/webhook"
    })

@app.route('/health')
def health():
    """Health check endpoint for Railway"""
    log_and_print("DEBUG", "Health check accessed")
    return jsonify({"status": "healthy"})

@app.route('/images/<path:path>')
def serve_images(path):
    """Serve images from the images directory"""
    log_and_print("DEBUG", f"Serving image: {path}")
    directory, filename = os.path.split(path)
    return send_from_directory(os.path.join('images', directory), filename)

@app.route('/test_message', methods=['POST'])
def test_message():
    """Test endpoint to simulate a message"""
    try:
        data = request.json
        phone = data.get('phone', '')
        message = data.get('message', '')
        
        if not phone or not message:
            return jsonify({"status": "error", "message": "Phone and message are required"}), 400
        
        # Process the test message directly
        handle_text_message(phone, message)
        return jsonify({"status": "success", "message": "Test message processed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages via WaAPI webhook"""
    try:
        # Log entire raw request for debugging
        print(f"[DEBUG] Raw webhook request: {request.data.decode('utf-8')[:500]}...")
        
        # Extract the webhook data
        webhook_data = request.json
        print(f"[DEBUG] Webhook received: {json.dumps(webhook_data)[:500]}...")
        
        # Check if this is a message we should process
        event_type = webhook_data.get('event', '')
        if event_type not in ['message', 'message_create']:
            print(f"[DEBUG] Skipping non-message event: {event_type}")
            return jsonify({"status": "success", "message": f"{event_type} event acknowledged"})
        
        # Extract message data
        message_data = webhook_data.get('data', {}).get('message', {})
        
        # Check if this is the bot's own message by checking fromMe flag
        is_from_me = False
        if message_data.get('fromMe') == True:
            is_from_me = True
        elif '_data' in message_data and message_data['_data'].get('fromMe') == True:
            is_from_me = True
            
        # If message is from the bot itself, don't process it
        if is_from_me:
            print("[DEBUG] Skipping bot's own message")
            return jsonify({"status": "success", "message": "Bot message skipped"})
        
        # Extract message ID to prevent duplicate processing
        message_id = message_data.get('id', {})
        serialized_id = None
        
        if isinstance(message_id, dict):
            serialized_id = message_id.get('_serialized', '')
        elif isinstance(message_id, str):
            serialized_id = message_id
        elif '_data' in message_data and '_serialized' in message_data['_data'].get('id', {}):
            serialized_id = message_data['_data']['id']['_serialized']
            
        # Generate deterministic ID if none found
        if not serialized_id:
            from_num = message_data.get('from', '')
            timestamp = message_data.get('timestamp', datetime.now().timestamp())
            serialized_id = f"{from_num}_{timestamp}"
        
        # Skip if we've already processed this message
        if serialized_id and serialized_id in processed_messages:
            log_and_print("INFO", f"Skipping duplicate message: {serialized_id}")
            return jsonify({"status": "success", "message": "Duplicate message skipped"})
        
        # Mark as processed if it has an ID
        if serialized_id:
            processed_messages[serialized_id] = datetime.now().timestamp()
            print(f"[DEBUG] Added message {serialized_id} to processed cache")
            
            # Limit cache size by removing old entries (keep last 100)
            if len(processed_messages) > 100:
                oldest = sorted(processed_messages.items(), key=lambda x: x[1])[0][0]
                processed_messages.pop(oldest)
                print(f"[DEBUG] Removed oldest message {oldest} from cache")
        
        # Extract message details - safer parsing with fallbacks
        message_type = message_data.get('type', '')
        if not message_type and '_data' in message_data:
            message_type = message_data['_data'].get('type', '')
            
        from_number = message_data.get('from', '')
        if not from_number and '_data' in message_data:
            from_number = message_data['_data'].get('from', '')
        
        # Get participant for group messages
        participant = None
        if '_data' in message_data and message_data['_data'].get('participant'):
            participant = message_data['_data'].get('participant')
        elif message_data.get('participant'):
            participant = message_data.get('participant')
        
        # Handle None values properly when parsing body
        raw_body = message_data.get('body')
        if raw_body is None and '_data' in message_data:
            raw_body = message_data['_data'].get('body')
            
        body = raw_body.strip() if isinstance(raw_body, str) else ''
        
        # Check for media in multiple ways
        has_media = (
            message_data.get('hasMedia', False) or 
            message_type in ['image', 'sticker', 'video', 'document'] or
            'mediaData' in message_data or
            ('_data' in message_data and 
             message_data['_data'].get('type') in ['image', 'sticker', 'video', 'document'])
        )
        
        log_and_print("INFO", f"Received message from {from_number}: {body}, Media: {has_media}, Type: {message_type}")
        if participant:
            print(f"[DEBUG] Message from participant: {participant} in group {from_number}")
        
        # Check if the number is a valid format
        if not ('@c.us' in str(from_number) or '@g.us' in str(from_number)):
            log_and_print("WARNING", f"Invalid chat format: {from_number}, trying to fix")
            # Try to fix the format - assume direct message
            from_number = ''.join(filter(str.isdigit, str(from_number))) + '@c.us'
            print(f"[DEBUG] Fixed chat format to: {from_number}")
        
        # Try to get media data in multiple ways
        media_data = webhook_data.get('data', {}).get('media', {})
        if not media_data:
            print("[DEBUG] No direct media data found, checking alternative locations")
            if '_data' in message_data and message_data['_data'].get('type') in ['image', 'sticker', 'video', 'document']:
                print("[DEBUG] Found media in _data")
                # Try to extract media from message_data itself for different webhook formats
                if hasattr(message_data, 'mediaData') or 'mediaData' in message_data:
                    media_data = message_data.get('mediaData', {})
                elif hasattr(message_data, 'mimetype') or 'mimetype' in message_data:
                    # It might be directly in the message
                    media_data = message_data
        
        # Handle media messages
        if has_media:
            log_and_print("INFO", f"Detected media in message from {from_number}, type: {message_type}")
            if not media_data:
                # If webhook doesn't include media data, try to fetch it
                log_and_print("INFO", "Media indicated but no media data in webhook, will attempt download")
            
            # Process as image regardless of empty media data - handler will check session state
            handle_image_message(from_number, media_data, participant)
            return jsonify({"status": "success", "message": "Media processed"})
            
        # Handle text messages
        handle_text_message(from_number, body, participant)
        return jsonify({"status": "success", "message": "Text processed"})
        
    except Exception as e:
        error_message = str(e)
        log_and_print("ERROR", f"Webhook error: {error_message}")
        print(f"[ERROR] Exception in webhook: {error_message}")
        traceback.print_exc()  # Print full stack trace
        return jsonify({"status": "error", "message": error_message})

###################
# MAIN APPLICATION
###################

if __name__ == '__main__':
    # Get port from environment variable (Railway provides this)
    port = int(os.getenv('PORT', 5000))
    
    # Print environment info for debugging
    print("-" * 50)
    print("MARKETING BOT STARTUP")
    print("-" * 50)
    print(f"Python version: {os.sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Environment variables: PORT={port}")
    print(f"OpenAI API Key Present: {'Yes' if OPENAI_API_KEY else 'No'}")
    print(f"WAAPI Token Present: {'Yes' if WAAPI_API_TOKEN else 'No'}")
    print(f"WAAPI Instance ID: {WAAPI_INSTANCE_ID}")
    print(f"Checking directories...")
    
    # Make sure directories exist and are writable
    for directory in ['images', 'images/input', 'images/output']:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory '{directory}' exists: {os.path.exists(directory)}")
    
    # Test PIL import
    print(f"PIL/Pillow version: {Image.__version__ if hasattr(Image, '__version__') else 'unknown'}")
    
    # Log startup
    log_and_print("INFO", f"Starting Marketing Bot API (WhatsApp Version) on port {port}")
    print("-" * 50)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False for production