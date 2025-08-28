import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# NEW IMPORTS for Stage 2
import sqlite3
import json
import re
from typing import Dict, List, Optional, Tuple

# -------------------------
# App & Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("belive-alps")
app = Flask(__name__)
CORS(app)

# -------------------------
# Globals
# -------------------------
model = None
feature_names = None
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", os.path.dirname(__file__))

MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "best_rf_model.pkl")
FEATURES_FILENAME = os.environ.get("FEATURES_FILENAME", "feature_names.pkl")

# -------------------------
# ALPS Stage 2 Classes (NEW)
# -------------------------

class DataDrivenALPSAnalyzer:
    """
    ALPS Stage 2 analyzer based on your actual lead success patterns:
    - 64.22% overall success rate
    - High urgency (30 days) = 68.82% success
    - Medium urgency (90 days) = 68.79% success  
    - Higher budget (RM1500+) = 68.65% success
    - Viewing arrangement intent = 63.83% success
    """
    
    def __init__(self):
        # Based on your data: timeline urgency is the #1 predictor
        self.timeline_signals = {
            'high_urgency': {  # Within 30 days = 68.82% success rate
                'patterns': [
                    r'next.*week', r'this.*week', r'within.*week',
                    r'by.*next.*\w+day', r'urgent', r'asap', r'immediately',
                    r'need.*soon', r'move.*in.*soon', r'available.*when'
                ],
                'weight': 12,  # Highest weight - strongest predictor
                'success_rate': 68.82,
                'description': 'High urgency timeline (within 30 days)'
            },
            
            'medium_urgency': {  # Within 90 days = 68.79% success rate
                'patterns': [
                    r'next.*month', r'this.*month', r'within.*\d+.*month',
                    r'by.*\w+uary|by.*\w+ember', r'end.*of.*month',
                    r'move.*in.*\d+.*month', r'looking.*for.*\w+.*month'
                ],
                'weight': 11,
                'success_rate': 68.79,
                'description': 'Medium urgency timeline (within 90 days)'
            },
            
            'vague_timeline': {  # Distant future = lower success
                'patterns': [
                    r'maybe', r'sometime', r'future', r'eventually',
                    r'not.*sure.*when', r'no.*rush', r'flexible.*timing',
                    r'next.*year', r'later.*this.*year'
                ],
                'weight': -8,  # Strong negative signal
                'success_rate': 52.94,
                'description': 'Vague or distant timeline'
            }
        }
        
        # Budget signals - higher budget = better success (68.65% for RM1500+)
        self.budget_signals = {
            'high_budget': {  # RM1500+ = 68.65% success
                'patterns': [
                    r'rm\s*1[5-9]\d{2}', r'rm\s*[2-9]\d{3}', r'\$[2-9]\d{2}',
                    r'budget.*1[5-9]\d{2}', r'afford.*1[5-9]\d{2}', r'up.*to.*1[5-9]\d{2}',
                    r'2\d{3}.*month', r'[15-20]\d{2}.*monthly'
                ],
                'weight': 10,
                'success_rate': 68.65,
                'description': 'High budget range (RM1500+)'
            },
            
            'medium_budget': {  # RM500-1000 = 63.43% success
                'patterns': [
                    r'rm\s*[5-9]\d{2}', r'rm\s*1[0-4]\d{2}', r'\$[5-9]\d{2}',
                    r'budget.*[5-9]\d{2}', r'afford.*[5-9]\d{2}',
                    r'[5-9]\d{2}.*month', r'1[0-4]\d{2}.*month'
                ],
                'weight': 6,
                'success_rate': 63.43,
                'description': 'Medium budget range (RM500-1500)'
            },
            
            'budget_concerns': {  # Price sensitivity
                'patterns': [
                    r'cheap', r'cheaper', r'lowest.*price', r'discount',
                    r'too.*expensive', r'cannot.*afford', r'tight.*budget',
                    r'student.*budget', r'minimal.*budget'
                ],
                'weight': -7,
                'success_rate': 25.0,  # Very low success for under RM500
                'description': 'Budget concerns or price sensitivity'
            }
        }
        
        # Intent signals - viewing arrangement = 63.83% success
        self.intent_signals = {
            'viewing_intent': {  # Strong action intent
                'patterns': [
                    r'schedule.*view', r'arrange.*view', r'visit.*property',
                    r'see.*property', r'tour', r'inspection', r'viewing',
                    r'can.*i.*see', r'when.*can.*visit', r'available.*to.*view'
                ],
                'weight': 15,  # Very high weight - direct action
                'success_rate': 63.83,
                'description': 'Wants to arrange property viewing'
            },
            
            'room_selection': {  # Room selection intent = 63.64% success
                'patterns': [
                    r'queen.*room', r'single.*room', r'studio', r'private.*room',
                    r'shared.*room', r'master.*room', r'bedroom.*prefer',
                    r'type.*of.*room', r'room.*available'
                ],
                'weight': 8,
                'success_rate': 63.64,
                'description': 'Specific room type preferences'
            },
            
            'application_ready': {  # High commitment
                'patterns': [
                    r'application', r'apply', r'book', r'reserve', r'confirm',
                    r'deposit', r'contract', r'agreement', r'paperwork',
                    r'ready.*to.*sign', r'documents.*ready'
                ],
                'weight': 18,  # Highest weight - ready to commit
                'success_rate': 80.0,  # Estimated high success
                'description': 'Ready for application/booking'
            },
            
            'just_browsing': {  # Low commitment
                'patterns': [
                    r'just.*looking', r'just.*browsing', r'just.*checking',
                    r'general.*info', r'curious', r'window.*shopping',
                    r'getting.*idea', r'exploring.*option'
                ],
                'weight': -10,
                'success_rate': 30.0,  # Estimated low success
                'description': 'Just browsing, low commitment'
            }
        }
        
        # Property-specific signals from your room types
        self.property_signals = {
            'specific_requirements': {
                'patterns': [
                    r'parking', r'furnished', r'aircon', r'wifi', r'laundry',
                    r'kitchen', r'bathroom', r'balcony', r'near.*mrt', r'near.*lrt',
                    r'transport', r'location', r'facilities', r'amenities'
                ],
                'weight': 7,
                'success_rate': 65.0,  # Estimated based on specificity
                'description': 'Specific property requirements'
            },
            
            'group_size_clarity': {  # Based on your No_of_Pax analysis
                'patterns': [
                    r'for.*\d+.*people', r'we.*are.*\d+', r'\d+.*person',
                    r'couple', r'single.*person', r'sharing.*with',
                    r'me.*and.*my', r'group.*of.*\d+'
                ],
                'weight': 5,
                'success_rate': 64.0,  # Average from your group size data
                'description': 'Clear about group size/occupancy'
            }
        }
        
        # Qualification signals 
        self.qualification_signals = {
            'employment_stable': {
                'patterns': [
                    r'working', r'employed', r'job', r'salary', r'income',
                    r'company', r'office', r'professional', r'stable.*income',
                    r'full.*time', r'permanent.*position'
                ],
                'weight': 9,
                'success_rate': 70.0,  # Employment indicates ability to pay
                'description': 'Stable employment/income'
            },
            
            'student_status': {  # Students typically have budget constraints
                'patterns': [
                    r'student', r'studying', r'university', r'college',
                    r'intern', r'scholarship', r'allowance'
                ],
                'weight': -5,
                'success_rate': 45.0,  # Students often have budget issues
                'description': 'Student status (budget constraints)'
            },
            
            'references_ready': {
                'patterns': [
                    r'reference', r'guarantor', r'documents.*ready',
                    r'proof.*income', r'bank.*statement', r'employer.*letter'
                ],
                'weight': 12,
                'success_rate': 75.0,  # Strong qualification indicator
                'description': 'Has references/documentation ready'
            }
        }
    
    def analyze_message(self, message: str, conversation_context: Dict = None) -> Dict:
        """
        Analyze a message using data-driven patterns from your lead success data
        """
        message_clean = message.lower().strip()
        
        analysis = {
            'message': message,
            'total_score_impact': 0,
            'confidence_level': 'medium',
            'detected_signals': [],
            'success_indicators': [],
            'risk_factors': [],
            'recommendations': []
        }
        
        # Analyze all signal categories
        signal_categories = [
            ('timeline', self.timeline_signals),
            ('budget', self.budget_signals), 
            ('intent', self.intent_signals),
            ('property', self.property_signals),
            ('qualification', self.qualification_signals)
        ]
        
        for category_name, signals in signal_categories:
            category_score = 0
            category_signals = []
            
            for signal_name, signal_data in signals.items():
                matches = 0
                matched_patterns = []
                
                for pattern in signal_data['patterns']:
                    pattern_matches = re.findall(pattern, message_clean)
                    if pattern_matches:
                        matches += len(pattern_matches)
                        matched_patterns.extend(pattern_matches)
                
                if matches > 0:
                    # Calculate impact based on your data
                    signal_impact = min(matches * signal_data['weight'], abs(signal_data['weight']) * 1.5)
                    category_score += signal_impact
                    
                    signal_detail = {
                        'category': category_name,
                        'signal': signal_name,
                        'description': signal_data['description'],
                        'matches': matches,
                        'weight': signal_data['weight'],
                        'impact': signal_impact,
                        'success_rate': signal_data['success_rate'],
                        'matched_text': matched_patterns[:3]  # First 3 matches
                    }
                    
                    analysis['detected_signals'].append(signal_detail)
                    category_signals.append(signal_detail)
                    
                    # Categorize as success indicator or risk factor
                    if signal_impact > 0:
                        analysis['success_indicators'].append(signal_detail)
                    else:
                        analysis['risk_factors'].append(signal_detail)
            
            analysis[f'{category_name}_score'] = category_score
            analysis['total_score_impact'] += category_score
        
        # Generate confidence level based on signal strength
        if analysis['total_score_impact'] > 15:
            analysis['confidence_level'] = 'high'
        elif analysis['total_score_impact'] < -10:
            analysis['confidence_level'] = 'low'
        else:
            analysis['confidence_level'] = 'medium'
        
        # Generate recommendations based on detected patterns
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Timeline-based recommendations
        timeline_signals = [s for s in analysis['detected_signals'] if s['category'] == 'timeline']
        if any(s['signal'] == 'high_urgency' for s in timeline_signals):
            recommendations.append("HIGH PRIORITY: Customer has urgent timeline - assign to agent immediately")
        elif any(s['signal'] == 'vague_timeline' for s in timeline_signals):
            recommendations.append("Follow up needed: Customer has unclear timeline - bot should probe for specifics")
        
        # Intent-based recommendations  
        intent_signals = [s for s in analysis['detected_signals'] if s['category'] == 'intent']
        if any(s['signal'] == 'viewing_intent' for s in intent_signals):
            recommendations.append("READY TO CONVERT: Customer wants viewing - agent should prioritize scheduling")
        elif any(s['signal'] == 'application_ready' for s in intent_signals):
            recommendations.append("HOT LEAD: Customer ready to apply - immediate agent handoff required")
        elif any(s['signal'] == 'just_browsing' for s in intent_signals):
            recommendations.append("NURTURE LEAD: Customer just browsing - bot should educate and build interest")
        
        # Budget-based recommendations
        budget_signals = [s for s in analysis['detected_signals'] if s['category'] == 'budget']
        if any(s['signal'] == 'high_budget' for s in budget_signals):
            recommendations.append("QUALIFIED LEAD: High budget range - prioritize premium properties")
        elif any(s['signal'] == 'budget_concerns' for s in budget_signals):
            recommendations.append("PRICE SENSITIVE: Focus on value proposition and budget options")
        
        if not recommendations:
            recommendations.append("Standard follow-up: Continue conversation to gather more qualifying information")
        
        return recommendations
    
    def calculate_updated_score(self, current_score: float, message_analysis: Dict, 
                              conversation_length: int = 1) -> float:
        """
        Calculate updated ALPS score based on message analysis
        Uses dampening and recency weighting based on your success data
        """
        # Base score impact
        raw_impact = message_analysis['total_score_impact']
        
        # Apply recency weighting - later messages in conversation matter more
        recency_weight = min(1.0, 0.5 + (conversation_length * 0.1))
        
        # Apply dampening to prevent wild swings - based on your data variance
        dampening_factor = 0.25  # Conservative approach
        
        # Calculate weighted impact
        weighted_impact = raw_impact * recency_weight * dampening_factor
        
        # Confidence-based adjustment
        confidence_multiplier = {
            'high': 1.2,    # Boost high-confidence signals
            'medium': 1.0,  # Standard weighting
            'low': 0.8      # Reduce low-confidence signals
        }
        
        final_impact = weighted_impact * confidence_multiplier.get(
            message_analysis['confidence_level'], 1.0
        )
        
        # Calculate new score with bounds
        new_score = max(0, min(100, current_score + final_impact))
        
        return round(new_score, 2)
    
    def should_route_to_agent(self, score: float, message_analysis: Dict, 
                            threshold: float = 70.0) -> Tuple[bool, str]:
        """
        Determine if lead should be routed to agent based on score and signals
        """
        # Score-based routing
        if score >= threshold:
            return True, f"Score {score} exceeds threshold {threshold}"
        
        # Signal-based override routing (regardless of score)
        priority_signals = [
            'viewing_intent', 'application_ready', 'high_urgency'
        ]
        
        for signal_detail in message_analysis['detected_signals']:
            if signal_detail['signal'] in priority_signals:
                return True, f"Priority signal detected: {signal_detail['description']}"
        
        # Check for high-value combinations
        has_budget = any(s['category'] == 'budget' and s['impact'] > 0 
                        for s in message_analysis['detected_signals'])
        has_timeline = any(s['category'] == 'timeline' and s['impact'] > 0 
                          for s in message_analysis['detected_signals'])
        
        if has_budget and has_timeline and score > 60:
            return True, "High-value combination: budget + timeline clarity"
        
        return False, f"Score {score} below threshold {threshold}, no priority signals"

class DatabaseALPSIntegration:
    """
    ALPS Stage 2 system that integrates with your existing SQLite database
    Uses your chats/messages tables and extends with ALPS tracking
    """
    
    def __init__(self, db_path: str = "belive.db"):
        self.db_path = db_path
        self.analyzer = None  # Will be set from your DataDrivenALPSAnalyzer
    
    def set_analyzer(self, analyzer):
        """Set the ALPS analyzer instance"""
        self.analyzer = analyzer
    
    def create_or_get_chat(self, customer_phone: str, initial_data: Dict = None) -> int:
        """
        Create or get existing chat_id for a customer
        Uses your existing chats table structure
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Try to find existing customer
            customer_row = conn.execute(
                "SELECT customer_id FROM customers WHERE phone_e164 = ?", 
                (customer_phone,)
            ).fetchone()
            
            if customer_row:
                customer_id = customer_row[0]
            else:
                # Create new customer
                cursor = conn.execute(
                    "INSERT INTO customers (phone_e164) VALUES (?)",
                    (customer_phone,)
                )
                customer_id = cursor.lastrowid
            
            # Create new chat with initial data
            chat_data = initial_data or {}
            cursor = conn.execute("""
                INSERT INTO chats (
                    customer_id, customer_phone_number, budget, room_type,
                    nationality, gender, transportation, parking, no_of_pax,
                    lead_source, initial_contact_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                customer_id,
                customer_phone,
                chat_data.get('budget'),
                chat_data.get('room_type'),
                chat_data.get('nationality'),
                chat_data.get('gender'),
                chat_data.get('transportation'),
                chat_data.get('parking'),
                chat_data.get('no_of_pax'),
                chat_data.get('lead_source', 'Website'),
                datetime.now().date()
            ))
            
            chat_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Created chat {chat_id} for customer {customer_id} ({customer_phone})")
            return chat_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create/get chat: {e}")
            raise
        finally:
            conn.close()
    
    def start_alps_conversation(self, chat_id: int, initial_score: float, 
                               threshold: float = 70.0) -> Dict:
        """
        Start ALPS conversation tracking for an existing chat
        """
        if not self.analyzer:
            raise ValueError("ALPS analyzer not set. Call set_analyzer() first.")
        
        conn = sqlite3.connect(self.db_path)
        try:
            # Check if ALPS conversation already exists
            existing = conn.execute(
                "SELECT conversation_id FROM alps_conversations WHERE chat_id = ?", 
                (chat_id,)
            ).fetchone()
            
            if existing:
                # Return existing conversation
                conv_data = conn.execute("""
                    SELECT conversation_id, current_score, current_handler, threshold
                    FROM alps_conversations WHERE chat_id = ?
                """, (chat_id,)).fetchone()
                
                return {
                    'conversation_id': conv_data[0],
                    'chat_id': chat_id,
                    'current_score': conv_data[1],
                    'assigned_to': conv_data[2],
                    'threshold': conv_data[3],
                    'status': 'existing'
                }
            
            # Get customer_id from chat
            chat_info = conn.execute(
                "SELECT customer_id FROM chats WHERE chat_id = ?", 
                (chat_id,)
            ).fetchone()
            
            if not chat_info:
                raise ValueError(f"Chat {chat_id} not found")
            
            customer_id = chat_info[0]
            
            # Determine initial handler
            initial_handler = 'agent' if initial_score >= threshold else 'bot'
            
            # Insert ALPS conversation
            cursor = conn.execute("""
                INSERT INTO alps_conversations 
                (chat_id, customer_id, initial_score, current_score, threshold, current_handler)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chat_id, customer_id, initial_score, initial_score, threshold, initial_handler))
            
            conversation_id = cursor.lastrowid
            
            # Insert initial score history
            conn.execute("""
                INSERT INTO alps_score_history 
                (conversation_id, new_score, trigger_type, confidence_level)
                VALUES (?, ?, 'initial', 'high')
            """, (conversation_id, initial_score))
            
            conn.commit()
            
            logger.info(f"Started ALPS conversation {conversation_id} for chat {chat_id}, score: {initial_score}, assigned to: {initial_handler}")
            
            return {
                'conversation_id': conversation_id,
                'chat_id': chat_id,
                'initial_score': initial_score,
                'current_score': initial_score,
                'assigned_to': initial_handler,
                'threshold': threshold,
                'status': 'created'
            }
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to start ALPS conversation: {e}")
            raise
        finally:
            conn.close()
    
    def add_message(self, chat_id: int, text: str, sender_type: str) -> int:
        """
        Add message to your existing messages table
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                INSERT INTO messages (chat_id, sender_type, sent_at, text, char_len)
                VALUES (?, ?, ?, ?, ?)
            """, (
                chat_id,
                sender_type,
                datetime.now(),
                text,
                len(text)
            ))
            
            message_id = cursor.lastrowid
            conn.commit()
            
            return message_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add message: {e}")
            raise
        finally:
            conn.close()
    
    def process_message_with_alps(self, chat_id: int, text: str, 
                                 sender_type: str = 'customer') -> Dict:
        """
        Main method: Add message and update ALPS score
        """
        if not self.analyzer:
            raise ValueError("ALPS analyzer not set")
        
        # Add message to database first
        message_id = self.add_message(chat_id, text, sender_type)
        
        # Only analyze customer messages for scoring
        if sender_type != 'customer':
            return {
                'message_id': message_id,
                'chat_id': chat_id,
                'score_change': 0,
                'routing_action': None,
                'reason': 'Non-customer message'
            }
        
        conn = sqlite3.connect(self.db_path)
        try:
            # Get ALPS conversation
            conv_data = conn.execute("""
                SELECT conversation_id, current_score, threshold, current_handler
                FROM alps_conversations WHERE chat_id = ?
            """, (chat_id,)).fetchone()
            
            if not conv_data:
                # No ALPS conversation exists - return basic info
                return {
                    'message_id': message_id,
                    'chat_id': chat_id,
                    'score_change': 0,
                    'routing_action': None,
                    'reason': 'No ALPS conversation tracking'
                }
            
            conversation_id, current_score, threshold, current_handler = conv_data
            
            # Analyze message
            analysis = self.analyzer.analyze_message(text)
            
            # Get conversation length for context
            conversation_length = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE chat_id = ? AND sender_type = 'customer'",
                (chat_id,)
            ).fetchone()[0]
            
            # Calculate new score
            new_score = self.analyzer.calculate_updated_score(
                current_score, analysis, conversation_length
            )
            
            score_change = new_score - current_score
            
            # Update conversation score
            conn.execute("""
                UPDATE alps_conversations 
                SET current_score = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE conversation_id = ?
            """, (new_score, conversation_id))
            
            # Add score history if significant change
            if abs(score_change) > 1.0:
                conn.execute("""
                    INSERT INTO alps_score_history 
                    (conversation_id, message_id, old_score, new_score, score_change, 
                     trigger_type, confidence_level)
                    VALUES (?, ?, ?, ?, ?, 'message_analysis', ?)
                """, (
                    conversation_id, message_id, current_score, new_score, 
                    score_change, analysis['confidence_level']
                ))
            
            # Store signal analysis details
            for signal in analysis['detected_signals']:
                conn.execute("""
                    INSERT INTO alps_signal_analysis 
                    (message_id, conversation_id, signal_category, signal_name, 
                     signal_description, matches_found, matched_text, weight_applied, 
                     score_impact, success_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message_id, conversation_id, signal['category'], signal['signal'],
                    signal['description'], signal['matches'], 
                    json.dumps(signal['matched_text']), signal['weight'],
                    signal['impact'], signal['success_rate']
                ))
            
            # Check for routing changes
            should_route, route_reason = self.analyzer.should_route_to_agent(
                new_score, analysis, threshold
            )
            
            routing_action = None
            new_handler = current_handler
            
            if should_route and current_handler == 'bot':
                new_handler = 'agent'
                routing_action = 'bot_to_agent'
            elif not should_route and current_handler == 'agent' and new_score < threshold - 5:
                new_handler = 'bot'
                routing_action = 'agent_to_bot'
            
            # Log routing action if needed
            if routing_action:
                conn.execute("""
                    INSERT INTO alps_routing_actions 
                    (conversation_id, message_id, old_handler, new_handler, 
                     routing_reason, score_at_routing, threshold_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    conversation_id, message_id, current_handler, new_handler,
                    route_reason, new_score, threshold
                ))
                
                # Update conversation handler
                conn.execute("""
                    UPDATE alps_conversations 
                    SET current_handler = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE conversation_id = ?
                """, (new_handler, conversation_id))
                
                logger.info(f"Routing action: {routing_action} for conversation {conversation_id}, score: {new_score}")
            
            conn.commit()
            
            return {
                'message_id': message_id,
                'conversation_id': conversation_id,
                'chat_id': chat_id,
                'old_score': round(current_score, 2),
                'new_score': round(new_score, 2),
                'score_change': round(score_change, 2),
                'current_handler': new_handler,
                'routing_action': routing_action,
                'routing_reason': route_reason if routing_action else None,
                'confidence': analysis['confidence_level'],
                'signals_detected': len(analysis['detected_signals']),
                'recommendations': analysis['recommendations'][:2]  # Top 2
            }
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to process message with ALPS: {e}")
            raise
        finally:
            conn.close()
    
    def get_conversation_summary(self, chat_id: int) -> Dict:
        """Get comprehensive conversation summary"""
        conn = sqlite3.connect(self.db_path)
        try:
            # Get conversation data with chat info
            conv_data = conn.execute("""
                SELECT 
                    ac.conversation_id, ac.initial_score, ac.current_score, 
                    ac.current_handler, ac.threshold, ac.status,
                    c.customer_phone_number, c.budget, c.nationality, c.room_type,
                    c.lead_source, ac.created_at
                FROM alps_conversations ac
                JOIN chats c ON c.chat_id = ac.chat_id
                WHERE ac.chat_id = ?
            """, (chat_id,)).fetchone()
            
            if not conv_data:
                return {'error': f'No ALPS conversation found for chat {chat_id}'}
            
            (conversation_id, initial_score, current_score, current_handler, threshold, 
             status, phone, budget, nationality, room_type, lead_source, created_at) = conv_data
            
            # Get message counts
            message_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_messages,
                    COUNT(CASE WHEN sender_type = 'customer' THEN 1 END) as customer_messages,
                    COUNT(CASE WHEN sender_type = 'agent' THEN 1 END) as agent_messages,
                    COUNT(CASE WHEN sender_type = 'bot' THEN 1 END) as bot_messages
                FROM messages WHERE chat_id = ?
            """, (chat_id,)).fetchone()
            
            # Get score history
            score_history = conn.execute("""
                SELECT new_score, score_change, trigger_type, created_at
                FROM alps_score_history 
                WHERE conversation_id = ? 
                ORDER BY created_at DESC LIMIT 10
            """, (conversation_id,)).fetchall()
            
            # Get top signals
            top_signals = conn.execute("""
                SELECT signal_name, signal_category, COUNT(*) as frequency,
                       AVG(score_impact) as avg_impact
                FROM alps_signal_analysis 
                WHERE conversation_id = ? 
                GROUP BY signal_name, signal_category
                ORDER BY frequency DESC, avg_impact DESC
                LIMIT 5
            """, (conversation_id,)).fetchall()
            
            # Get routing history
            routing_history = conn.execute("""
                SELECT old_handler, new_handler, routing_reason, score_at_routing, created_at
                FROM alps_routing_actions 
                WHERE conversation_id = ? 
                ORDER BY created_at DESC
            """, (conversation_id,)).fetchall()
            
            return {
                'conversation_id': conversation_id,
                'chat_id': chat_id,
                'customer_info': {
                    'phone': phone,
                    'budget': budget,
                    'nationality': nationality,
                    'room_type': room_type,
                    'lead_source': lead_source
                },
                'scores': {
                    'initial': initial_score,
                    'current': current_score,
                    'improvement': current_score - initial_score,
                    'threshold': threshold
                },
                'status': {
                    'current_handler': current_handler,
                    'status': status,
                    'created_at': created_at
                },
                'message_stats': {
                    'total': message_stats[0],
                    'customer': message_stats[1],
                    'agent': message_stats[2],
                    'bot': message_stats[3]
                },
                'score_history': [
                    {'score': row[0], 'change': row[1], 'trigger': row[2], 'time': row[3]}
                    for row in score_history
                ],
                'top_signals': [
                    {'signal': row[0], 'category': row[1], 'frequency': row[2], 'avg_impact': row[3]}
                    for row in top_signals
                ],
                'routing_history': [
                    {'from': row[0], 'to': row[1], 'reason': row[2], 'score': row[3], 'time': row[4]}
                    for row in routing_history
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation summary: {e}")
            return {'error': str(e)}
        finally:
            conn.close()
    
    def list_active_conversations(self, limit: int = 50) -> List[Dict]:
        """List active ALPS conversations with basic info"""
        conn = sqlite3.connect(self.db_path)
        try:
            conversations = conn.execute("""
                SELECT 
                    ac.conversation_id, ac.chat_id, ac.current_score, ac.current_handler,
                    c.customer_phone_number, c.budget, c.nationality,
                    COUNT(m.message_id) as message_count,
                    MAX(m.sent_at) as last_message_time,
                    ac.updated_at
                FROM alps_conversations ac
                JOIN chats c ON c.chat_id = ac.chat_id
                LEFT JOIN messages m ON m.chat_id = ac.chat_id
                WHERE ac.status = 'active'
                GROUP BY ac.conversation_id, ac.chat_id, ac.current_score, ac.current_handler,
                         c.customer_phone_number, c.budget, c.nationality, ac.updated_at
                ORDER BY ac.updated_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
            
            return [
                {
                    'conversation_id': row[0],
                    'chat_id': row[1],
                    'current_score': row[2],
                    'current_handler': row[3],
                    'customer_phone': row[4],
                    'budget': row[5],
                    'nationality': row[6],
                    'message_count': row[7],
                    'last_message_time': row[8],
                    'last_updated': row[9]
                }
                for row in conversations
            ]
            
        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            return []
        finally:
            conn.close()

# -------------------------
# Load artifacts on import (UNCHANGED)
# -------------------------
def load_model_artifacts():
    global model, feature_names
    model_path = os.path.join(ARTIFACT_DIR, MODEL_FILENAME)
    features_path = os.path.join(ARTIFACT_DIR, FEATURES_FILENAME)
    try:
        model = joblib.load(model_path)
        logger.info(f"✅ Loaded model: {type(model).__name__} from {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load model from {model_path}: {e}")
        model = None

    try:
        feature_names = joblib.load(features_path)
        if isinstance(feature_names, (list, np.ndarray, pd.Index)):
            feature_names = list(feature_names)
        logger.info(f"✅ Loaded feature_names: {len(feature_names)} from {features_path}")
    except Exception as e:
        logger.warning(f"⚠️ Could not load feature_names from {features_path}: {e}")
        feature_names = None

load_model_artifacts()

# -------------------------
# Initialize ALPS Stage 2 (NEW)
# -------------------------
analyzer = DataDrivenALPSAnalyzer()
db_alps = DatabaseALPSIntegration()
db_alps.set_analyzer(analyzer)

logger.info("✅ ALPS Stage 2 with database integration initialized")

# -------------------------
# Feature prep (UNCHANGED)
# -------------------------
def prepare_features(form_data):
    """
    Build a single-row DataFrame with the engineered / one-hot features
    your RandomForest was trained on.
    """
    current_date = datetime.now()
    feats = {}

    # Core numeric
    try:
        budget = float(form_data.get("budget", 800) or 800)
    except Exception:
        budget = 800.0
    feats["Budget"] = budget
    feats["Rental Proposed"] = budget

    pax_map = {"1 person": 1, "2 people": 2, "More than 2": 3}
    feats["No of Pax"] = pax_map.get(form_data.get("pax", "1 person"), 1)

    feats["contact_hour"] = current_date.hour
    feats["contact_month"] = current_date.month

    # Engineered
    if budget == 0:
        feats["Budget_Category_Encoded"] = 0
    elif budget < 500:
        feats["Budget_Category_Encoded"] = 1
    elif budget < 1000:
        feats["Budget_Category_Encoded"] = 2
    elif budget < 1500:
        feats["Budget_Category_Encoded"] = 3
    else:
        feats["Budget_Category_Encoded"] = 4

    feats["Move_Urgency_Encoded"] = 0
    movein = form_data.get("movein")
    if movein:
        try:
            move_dt = datetime.strptime(movein, "%Y-%m-%d")
            dd = (move_dt - current_date).days
            if dd <= 30:
                feats["Move_Urgency_Encoded"] = 1
            elif dd <= 90:
                feats["Move_Urgency_Encoded"] = 2
            else:
                feats["Move_Urgency_Encoded"] = 3
        except Exception:
            feats["Move_Urgency_Encoded"] = 0

    feats["Is_Weekend"] = 1 if current_date.weekday() >= 5 else 0
    feats["Is_Business_Hours"] = 1 if 9 <= current_date.hour <= 17 else 0

    # One-hot categories (prefixes used in your training)
    # Customer Journey
    journey_categories = [
        "Information_Collection","Property_Inquiry","Viewing_Arrangement",
        "Room_Selection","Property_Viewing","Booking_Process","Other","Unknown"
    ]
    cur_journey = "Unknown"
    for c in journey_categories:
        feats[f"Customer_Journey_Clean_{c}"] = 1 if c == cur_journey else 0

    # Gender
    gender_categories = ["Male","Female","Mixed","Unknown"]
    cur_gender = form_data.get("gender", "Unknown")
    if cur_gender not in gender_categories:
        cur_gender = "Unknown"
    for c in gender_categories:
        feats[f"Gender_Clean_{c}"] = 1 if c == cur_gender else 0

    # Lead Source
    lead_categories = [
        "Facebook","Google_Search","Google_Ads","Instagram","WhatsApp",
        "Website","Referral","Walk_In","Email","Phone_Call","Unknown"
    ]
    cur_source = "Website"
    for c in lead_categories:
        feats[f"Lead_Source_Standard_{c}"] = 1 if c == cur_source else 0

    # Room Type
    room_categories = ["Studio","1_Bedroom","2_Bedroom","3_Bedroom","Other","Unknown"]
    rt = str(form_data.get("room_type", "Unknown") or "Unknown")
    rtl = rt.lower()
    if "studio" in rtl:
        cur_room = "Studio"
    elif "1" in rt and "bed" in rtl:
        cur_room = "1_Bedroom"
    elif "2" in rt and "bed" in rtl:
        cur_room = "2_Bedroom"
    elif "3" in rt and "bed" in rtl:
        cur_room = "3_Bedroom"
    else:
        cur_room = "Unknown"
    for c in room_categories:
        feats[f"Room_Type_Standard_{c}"] = 1 if c == cur_room else 0

    # Transportation
    transport_categories = ["Car","Public Transport","Both","Unknown"]
    cur_transport = "Car" if form_data.get("car") == "Yes" else "Unknown"
    for c in transport_categories:
        feats[f"Transportation_{c}"] = 1 if c == cur_transport else 0

    # Parking
    parking_categories = ["Yes","No","Unknown"]
    cur_park = form_data.get("parking", "Unknown")
    for c in parking_categories:
        feats[f"Parking_{c}"] = 1 if c == cur_park else 0

    # Day of week
    weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    cur_wd = weekdays[current_date.weekday()]
    for d in weekdays:
        feats[f"contact_dayofweek_{d}"] = 1 if d == cur_wd else 0

    # Nationality (grouped)
    nationality_countries = [
        "Malaysia","Indonesia","India","Sudan","Zimbabwe","China",
        "Thailand","Myanmar","Pakistan","Yemen","Other"
    ]
    nat = form_data.get("nationality", "Other")
    if nat == "Others" and form_data.get("nationality_detail"):
        detail = str(form_data["nationality_detail"]).title()
        nat = detail if detail in nationality_countries else "Other"
    for c in nationality_countries:
        feats[f"Nationality_Standard_Grouped_{c}"] = 1 if c == nat else 0

    df = pd.DataFrame([feats])

    # Align to training feature order (very important!)
    if feature_names:
        # add any missing columns with 0
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        # drop any unexpected columns
        extra = [c for c in df.columns if c not in feature_names]
        if extra:
            df = df.drop(columns=extra)
        # reorder
        df = df[feature_names]

    # Ensure numeric dtypes
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df

# -------------------------
# Fallback score (UNCHANGED)
# -------------------------
def calculate_fallback_score(form_data):
    score = 50
    try:
        b = float(form_data.get("budget", 0) or 0)
        if b >= 1200: score += 20
        elif b >= 800: score += 15
        elif b >= 600: score += 10
    except Exception:
        pass

    if form_data.get("nationality") == "Malaysian":
        score += 15

    mv = form_data.get("movein")
    if mv:
        try:
            dd = (datetime.strptime(mv, "%Y-%m-%d") - datetime.now()).days
            if dd <= 30: score += 15
            elif dd <= 90: score += 10
        except Exception:
            pass

    if 9 <= datetime.now().hour <= 17:
        score += 10

    filled = sum(1 for v in form_data.values() if v not in (None, "", []))
    score += min(15, filled * 2)
    return max(0, min(100, score))

# -------------------------
# Routes - Stage 1 (UNCHANGED)
# -------------------------
@app.route("/")
def root():
    return jsonify({
        "status": "BeLive ALPS API running",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "features_loaded": feature_names is not None,
        "feature_count": len(feature_names) if feature_names else None,
        "artifact_dir": ARTIFACT_DIR,
        "model_filename": MODEL_FILENAME,
        "features_filename": FEATURES_FILENAME,
        "stage2_enabled": True,
        "active_conversations": len(db_alps.list_active_conversations()),
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/score", methods=["POST"])
def score():
    if not request.is_json:
        return jsonify({"error": "Send JSON", "score": 50}), 400
    payload = request.get_json(silent=True) or {}
    logger.info(f"Incoming keys: {list(payload.keys())}")

    # If model not loaded, return fallback (200 so the UI can proceed)
    if model is None:
        fb = calculate_fallback_score(payload)
        logger.error("Model not loaded — returning fallback")
        return jsonify({
            "score": fb,
            "timestamp": datetime.now().isoformat(),
            "model_used": False,
            "reason": "model_not_loaded"
        }), 200

    try:
        X = prepare_features(payload)
        logger.info(f"Prepared features shape: {X.shape}")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # assume binary with class 1 = success
            p1 = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
            score = max(0.0, min(100.0, p1 * 100.0))
        else:
            pred = model.predict(X)
            # if classify 0/1, treat as prob
            p1 = float(pred[0])
            score = max(0.0, min(100.0, p1 * 100.0))

        logger.info(
            f"[MODEL USED] type={type(model).__name__} "
            f"success_prob={success_probability:.4f} "
            f"score={score:.2f}"
        )

        return jsonify({
            'score': round(score, 2),
            'success_probability': round(success_probability, 4),
            'timestamp': datetime.now().isoformat(),
            'model_used': True,                     # <<<<<< add this
            'model_type': type(model).__name__      # <<<<<< and this
        })
    except Exception as e:
        logger.exception(f"Scoring failed, using fallback: {e}")
        fb = calculate_fallback_score(payload)
        return jsonify({
            "score": fb,
            "timestamp": datetime.now().isoformat(),
            "model_used": False,
            "reason": "exception",
            "error": str(e)
        }), 200

# -------------------------
# NEW ROUTES - Stage 2 with Database
# -------------------------

@app.route("/api/chat/start", methods=["POST"])
def start_chat_with_alps():
    """
    Start a new chat with ALPS tracking
    Combines Stage 1 scoring + Stage 2 conversation setup
    """
    if not request.is_json:
        return jsonify({"error": "Send JSON"}), 400
    
    data = request.get_json(silent=True) or {}
    customer_phone = data.get('customer_phone')
    
    if not customer_phone:
        return jsonify({"error": "customer_phone required"}), 400
    
    try:
        # Create or get chat
        chat_id = db_alps.create_or_get_chat(customer_phone, data)
        
        # Calculate Stage 1 score
        if model is not None:
            try:
                X = prepare_features(data)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    p1 = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
                    initial_score = max(0.0, min(100.0, p1 * 100.0))
                else:
                    pred = model.predict(X)
                    p1 = float(pred[0])
                    initial_score = max(0.0, min(100.0, p1 * 100.0))
                model_used = True
            except Exception as e:
                logger.warning(f"Stage 1 scoring failed: {e}")
                initial_score = calculate_fallback_score(data)
                model_used = False
        else:
            initial_score = calculate_fallback_score(data)
            model_used = False
        
        # Start ALPS conversation tracking
        threshold = float(data.get('threshold', 70.0))
        alps_result = db_alps.start_alps_conversation(chat_id, initial_score, threshold)
        
        logger.info(f"Started chat {chat_id} with ALPS score {initial_score}, assigned to {alps_result['assigned_to']}")
        
        return jsonify({
            'success': True,
            'chat_id': chat_id,
            'customer_phone': customer_phone,
            'stage1_score': round(initial_score, 2),
            'model_used': model_used,
            'alps_conversation': alps_result,
            'ready_for_messages': True,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to start chat with ALPS: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route("/api/chat/<int:chat_id>/message", methods=["POST"])
def send_message_with_alps(chat_id):
    """
    Send message and get real-time ALPS analysis
    Core Stage 2 functionality with database persistence
    """
    if not request.is_json:
        return jsonify({"error": "Send JSON"}), 400
    
    data = request.get_json(silent=True) or {}
    text = data.get('text')
    sender_type = data.get('sender_type', 'customer')
    
    if not text:
        return jsonify({"error": "text required"}), 400
    
    try:
        # Process message with ALPS analysis
        result = db_alps.process_message_with_alps(chat_id, text, sender_type)
        
        # Log significant events
        if result.get('routing_action'):
            logger.info(f"Chat {chat_id}: {result['routing_action']} - {result['routing_reason']}")
        
        if abs(result.get('score_change', 0)) > 3:
            logger.info(f"Chat {chat_id}: Significant score change {result['old_score']} → {result['new_score']}")
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            **result
        })
        
    except Exception as e:
        logger.error(f"Failed to send message with ALPS: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route("/api/chat/<int:chat_id>/status", methods=["GET"])
def get_chat_alps_status(chat_id):
    """Get comprehensive chat status with ALPS analysis"""
    try:
        summary = db_alps.get_conversation_summary(chat_id)
        
        if 'error' in summary:
            return jsonify({'success': False, 'error': summary['error']}), 404
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            **summary
        })
        
    except Exception as e:
        logger.error(f"Failed to get chat status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route("/api/chats/active", methods=["GET"])
def list_active_chats():
    """List all active chats with ALPS data"""
    try:
        limit = int(request.args.get('limit', 50))
        conversations = db_alps.list_active_conversations(limit)
        
        return jsonify({
            'success': True,
            'conversations': conversations,
            'total': len(conversations),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to list active chats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route("/api/demo/full_conversation", methods=["POST"])
def demo_full_conversation():
    """Demo a complete conversation flow with database persistence"""
    try:
        # Create demo customer
        demo_phone = f"+60123456{datetime.now().strftime('%H%M')}"
        demo_data = {
            'budget': 2200,
            'nationality': 'Malaysian',
            'room_type': 'Queen Private',
            'gender': 'Male',
            'no_of_pax': 1,
            'lead_source': 'Website Demo'
        }
        
        # Start chat with ALPS
        chat_id = db_alps.create_or_get_chat(demo_phone, demo_data)
        alps_result = db_alps.start_alps_conversation(chat_id, 65.0, 70.0)
        
        # Simulate conversation
        demo_messages = [
            ("Hi, I'm looking for a Queen Private room", "customer"),
            ("Hello! I'd be happy to help. What's your budget range?", "agent"),
            ("Around RM 2200 per month, I need to move in within 2 weeks", "customer"),
            ("Perfect! We have great options. Any specific requirements?", "agent"),
            ("I need parking and good internet. When can I schedule a viewing?", "customer"),
            ("I can arrange a viewing tomorrow. I have your details ready.", "agent"),
            ("Excellent! I have all my documents ready including income proof", "customer")
        ]
        
        conversation_log = []
        for text, sender_type in demo_messages:
            result = db_alps.process_message_with_alps(chat_id, text, sender_type)
            conversation_log.append({
                'text': text,
                'sender': sender_type,
                'result': {
                    'score': result.get('new_score'),
                    'change': result.get('score_change'),
                    'handler': result.get('current_handler'),
                    'routing_action': result.get('routing_action'),
                    'confidence': result.get('confidence')
                }
            })
        
        # Get final summary
        final_summary = db_alps.get_conversation_summary(chat_id)
        
        return jsonify({
            'success': True,
            'demo_info': {
                'chat_id': chat_id,
                'customer_phone': demo_phone,
                'initial_setup': alps_result
            },
            'conversation_log': conversation_log,
            'final_summary': final_summary,
            'database_persisted': True
        })
        
    except Exception as e:
        logger.error(f"Demo conversation failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route("/api/health")
def health():
    """Enhanced health check with database and ALPS status"""
    try:
        # Test database connection
        conn = sqlite3.connect(db_alps.db_path)
        active_conversations = conn.execute(
            "SELECT COUNT(*) FROM alps_conversations WHERE status = 'active'"
        ).fetchone()[0] if conn else 0
        conn.close()
        db_status = 'connected'
    except Exception as e:
        active_conversations = 0
        db_status = f'error: {str(e)}'
    
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "features_loaded": feature_names is not None,
        "feature_count": len(feature_names) if feature_names else None,
        "database_status": db_status,
        "active_alps_conversations": active_conversations,
        "alps_analyzer_ready": db_alps.analyzer is not None,
        "stage2_enabled": True,
        "timestamp": datetime.now().isoformat()
    })

# -------------------------
# Entrypoint (UPDATED)
# -------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting BeLive ALPS API with Stage 2 database integration")
    logger.info(f"📊 Database: {db_alps.db_path}")
    logger.info(f"🧠 ALPS analyzer ready: {db_alps.analyzer is not None}")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_ENV") == "development")
