"""
Conversation Manager for Disease Prediction Chatbot
Handles dialogue flow, context tracking, and conversational logic
"""

from typing import List, Dict, Optional
from enum import Enum


class ConversationState(Enum):
    """Conversation states"""
    STARTED = "started"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    PREDICTING = "predicting"
    FOLLOW_UP = "follow_up"
    ENDED = "ended"


class ConversationManager:
    """
    Manage conversation flow and context for symptom checker chatbot
    """
    
    def __init__(self):
        """Initialize conversation manager"""
        self.state = ConversationState.STARTED
        self.collected_symptoms = []
        self.unconfirmed_symptoms = []
        self.context = {}
        self.turn_count = 0
    
    def start_conversation(self) -> str:
        """
        Get initial greeting message
        
        Returns:
            Greeting message
        """
        self.state = ConversationState.COLLECTING
        self.turn_count = 0
        
        message = """
🏥 Welcome to the Disease Prediction Assistant!

⚠️  IMPORTANT DISCLAIMER:
This is an AI-powered tool for informational purposes only.
It is NOT a substitute for professional medical advice.
Always consult a qualified healthcare provider for diagnosis and treatment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I'm here to help analyze symptoms and suggest possible conditions.

Please describe how you're feeling or what symptoms you're experiencing.
You can type naturally - for example:
  • "I have a headache and feel dizzy"
  • "My stomach hurts and I feel nauseous"
  • "Running a fever with a cough"

Type 'help' for more options or 'exit' to quit.
        """
        return message.strip()
    
    def process_input(self, user_input: str, extracted_symptoms: List[str]) -> Dict:
        """
        Process user input and determine next action
        
        Args:
            user_input: Raw user input
            extracted_symptoms: Symptoms extracted from input
            
        Returns:
            Dictionary with action, message, and other data
        """
        self.turn_count += 1
        user_input = user_input.strip().lower()
        
        # Handle commands
        if user_input in ['exit', 'quit', 'bye']:
            return self._handle_exit()
        
        if user_input in ['help', '?']:
            return self._handle_help()
        
        if user_input in ['reset', 'restart', 'new']:
            return self._handle_reset()
        
        # Handle based on current state
        if self.state == ConversationState.COLLECTING:
            return self._handle_collecting(user_input, extracted_symptoms)
        
        elif self.state == ConversationState.CONFIRMING:
            return self._handle_confirming(user_input, extracted_symptoms)
        
        elif self.state == ConversationState.FOLLOW_UP:
            return self._handle_follow_up(user_input)
        
        else:
            return {
                'action': 'error',
                'message': "I'm not sure how to handle this. Type 'help' for options."
            }
    
    def _handle_collecting(self, user_input: str, extracted_symptoms: List[str]) -> Dict:
        """Handle symptom collection state"""
        
        if not extracted_symptoms:
            # No symptoms detected
            if self.turn_count == 1:
                # First turn, be more helpful
                message = """
I didn't catch any specific symptoms from that.

Could you describe your symptoms more specifically? For example:
  • Body parts affected: head, stomach, chest, etc.
  • Sensations: pain, ache, burning, tingling, etc.
  • Other symptoms: fever, cough, nausea, fatigue, etc.
"""
                return {
                    'action': 'clarify',
                    'message': message.strip()
                }
            else:
                message = "I didn't detect any new symptoms. Could you try describing them differently?"
                return {
                    'action': 'clarify',
                    'message': message
                }
        
        # Add new symptoms
        new_symptoms = [s for s in extracted_symptoms if s not in self.collected_symptoms]
        
        if new_symptoms:
            self.collected_symptoms.extend(new_symptoms)
            self.unconfirmed_symptoms = new_symptoms
            
            # Build response
            message = f"I understand you're experiencing:\n"
            for symptom in new_symptoms:
                symptom_display = symptom.replace('_', ' ').title()
                message += f"  • {symptom_display}\n"
            
            message += f"\nTotal symptoms collected: {len(self.collected_symptoms)}\n\n"
            
            if len(self.collected_symptoms) < 2:
                message += "Do you have any other symptoms? (Type 'yes' to add more, 'no' to get predictions)"
                return {
                    'action': 'ask_more',
                    'message': message,
                    'symptoms': self.collected_symptoms
                }
            else:
                message += "Would you like to:\n"
                message += "  1. Add more symptoms (type: more/yes)\n"
                message += "  2. Get predictions now (type: predict/no/done)\n"
                message += "  3. Review symptoms (type: review)"
                return {
                    'action': 'ready',
                    'message': message,
                    'symptoms': self.collected_symptoms
                }
        else:
            message = "You already mentioned those symptoms. Any other symptoms?"
            return {
                'action': 'ask_more',
                'message': message,
                'symptoms': self.collected_symptoms
            }
    
    def _handle_confirming(self, user_input: str, extracted_symptoms: List[str]) -> Dict:
        """Handle confirmation state"""
        
        if user_input in ['yes', 'y', 'more', 'add']:
            self.state = ConversationState.COLLECTING
            return {
                'action': 'continue_collecting',
                'message': "What other symptoms are you experiencing?"
            }
        
        elif user_input in ['no', 'n', 'done', 'predict', 'ready']:
            self.state = ConversationState.PREDICTING
            return {
                'action': 'predict',
                'message': "Analyzing your symptoms...",
                'symptoms': self.collected_symptoms
            }
        
        elif user_input in ['review', 'list', 'show']:
            return self._show_collected_symptoms()
        
        else:
            # Try to extract symptoms from response
            if extracted_symptoms:
                return self._handle_collecting(user_input, extracted_symptoms)
            else:
                message = "Please type 'yes' to add more symptoms, or 'no' to get predictions."
                return {
                    'action': 'clarify',
                    'message': message
                }
    
    def _handle_follow_up(self, user_input: str) -> Dict:
        """Handle follow-up state after predictions"""
        
        if user_input in ['new', 'restart', 'again']:
            return self._handle_reset()
        
        elif user_input in ['exit', 'quit', 'bye', 'no']:
            return self._handle_exit()
        
        else:
            message = """
What would you like to do?
  • Start a new consultation (type: new/restart)
  • Exit (type: exit/quit)
            """
            return {
                'action': 'clarify',
                'message': message.strip()
            }
    
    def _show_collected_symptoms(self) -> Dict:
        """Show all collected symptoms"""
        if not self.collected_symptoms:
            message = "No symptoms collected yet."
        else:
            message = f"Collected symptoms ({len(self.collected_symptoms)}):\n"
            for i, symptom in enumerate(self.collected_symptoms, 1):
                symptom_display = symptom.replace('_', ' ').title()
                message += f"  {i}. {symptom_display}\n"
        
        return {
            'action': 'info',
            'message': message,
            'symptoms': self.collected_symptoms
        }
    
    def _handle_help(self) -> Dict:
        """Handle help command"""
        message = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HELP - Available Commands
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Symptom Collection:
  • Just describe your symptoms naturally
  • Example: "I have a headache and feel dizzy"
  
Commands:
  help     - Show this help message
  review   - Show all collected symptoms
  more     - Add more symptoms
  predict  - Get predictions with current symptoms
  reset    - Start over with new consultation
  exit     - Quit the chatbot
  
Tips:
  • Be specific about your symptoms
  • Mention body parts: head, stomach, chest, etc.
  • Include sensations: pain, burning, tingling, etc.
  • Add context: fever, cough, fatigue, etc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        return {
            'action': 'info',
            'message': message.strip()
        }
    
    def _handle_reset(self) -> Dict:
        """Handle reset command"""
        self.collected_symptoms = []
        self.unconfirmed_symptoms = []
        self.state = ConversationState.COLLECTING
        self.turn_count = 0
        
        message = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Starting new consultation...

Please describe your symptoms.
        """
        return {
            'action': 'reset',
            'message': message.strip()
        }
    
    def _handle_exit(self) -> Dict:
        """Handle exit command"""
        self.state = ConversationState.ENDED
        
        message = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Thank you for using the Disease Prediction Assistant!

⚕️  Remember: Always consult a healthcare professional for proper
   diagnosis and treatment. This tool is for informational purposes only.

Stay healthy! 👋

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        return {
            'action': 'exit',
            'message': message.strip()
        }
    
    def transition_to_follow_up(self):
        """Transition to follow-up state after showing predictions"""
        self.state = ConversationState.FOLLOW_UP
    
    def get_state(self) -> ConversationState:
        """Get current conversation state"""
        return self.state
    
    def get_collected_symptoms(self) -> List[str]:
        """Get all collected symptoms"""
        return self.collected_symptoms.copy()


# Example usage
if __name__ == "__main__":
    manager = ConversationManager()
    
    # Test conversation flow
    print(manager.start_conversation())
    
    # Simulate collecting symptoms
    response = manager.process_input(
        "I have a headache",
        extracted_symptoms=['headache']
    )
    print(f"\n{response['message']}")
    
    response = manager.process_input(
        "yes, also dizzy",
        extracted_symptoms=['dizzy']
    )
    print(f"\n{response['message']}")
    
    response = manager.process_input(
        "no, that's all",
        extracted_symptoms=[]
    )
    print(f"\nAction: {response['action']}")
    print(response['message'])
