# ============================================================
# ai_assistant.py - Groq AI Lead Assistant (NO AUTH)
# ============================================================

import os
import json
import sqlite3
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class AILeadAssistant:
    def __init__(self, db_path='leads.db'):
        self.db_path = db_path
        self.api_key = os.environ.get('GROQ_API_KEY')
        self.model = 'llama-3.1-8b-instant'
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # Validate API key
        if not self.api_key:
            print("❌ GROQ_API_KEY not found in .env file")
            print("   Please add: GROQ_API_KEY=your_key_here")
            self.available = False
        elif not self.api_key.startswith('gsk_'):
            print(f"⚠️ Invalid API key format - should start with 'gsk_'")
            print(f"   Got: {self.api_key[:20]}...")
            self.available = False
        else:
            # Test the API key with a simple request
            try:
                test_response = requests.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 5
                    },
                    timeout=10
                )
                
                if test_response.status_code == 200:
                    self.available = True
                    print(f"✅ Groq API ready! (Model: {self.model})")
                else:
                    self.available = False
                    error_msg = test_response.json().get('error', {}).get('message', 'Unknown error')
                    print(f"❌ Groq API key invalid: {error_msg}")
                    print(f"   Status: {test_response.status_code}")
            except requests.exceptions.ConnectionError:
                self.available = False
                print("❌ Cannot connect to Groq API - Check your internet connection")
            except requests.exceptions.Timeout:
                self.available = False
                print("❌ Groq API timeout - Try again later")
            except Exception as e:
                self.available = False
                print(f"❌ Groq API error: {e}")
        
        # Store conversation history
        self.conversation_history = []
    
    def _get_db_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_all_leads_data(self, limit=500):
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        leads = cursor.execute('''
            SELECT 
                id,
                display_id,
                client_data,
                prediction,
                probability_yes,
                probability_no,
                confidence,
                priority,
                message,
                timestamp,
                source
            FROM leads 
            ORDER BY display_id ASC
            LIMIT ?
        ''', (limit,)).fetchall()
        
        conn.close()
        
        all_leads = []
        for lead in leads:
            try:
                client = json.loads(lead['client_data'])
            except:
                client = {}
            all_leads.append({
                'display_id': lead['display_id'],
                'id': lead['id'],
                'client': client,
                'prediction': lead['prediction'],
                'probability_yes': lead['probability_yes'],
                'probability_no': lead['probability_no'],
                'confidence': lead['confidence'],
                'priority': lead['priority'],
                'message': lead['message'],
                'timestamp': lead['timestamp'],
                'source': lead['source']
            })
        
        return all_leads
    
    def get_single_lead_full(self, display_id):
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        lead = cursor.execute('''
            SELECT 
                id,
                display_id,
                client_data,
                prediction,
                probability_yes,
                probability_no,
                confidence,
                priority,
                message,
                timestamp,
                source
            FROM leads 
            WHERE display_id = ?
        ''', (display_id,)).fetchone()
        
        conn.close()
        
        if not lead:
            return None
        
        try:
            client = json.loads(lead['client_data'])
        except:
            client = {}
        
        return {
            'display_id': lead['display_id'],
            'id': lead['id'],
            'client': client,
            'prediction': lead['prediction'],
            'probability_yes': lead['probability_yes'],
            'probability_no': lead['probability_no'],
            'confidence': lead['confidence'],
            'priority': lead['priority'],
            'message': lead['message'],
            'timestamp': lead['timestamp'],
            'source': lead['source']
        }
    
    def build_complete_context(self):
        all_leads = self.get_all_leads_data(limit=500)
        
        if not all_leads:
            return {'context': "No leads found.", 'leads': [], 'stats': {}, 'mode': 'empty'}
        
        context_parts = []
        
        total = len(all_leads)
        high = sum(1 for l in all_leads if l['prediction'] == 'Yes')
        low = total - high
        conv_rate = round((high / total * 100) if total > 0 else 0, 1)
        avg_conf = round(sum(l['confidence'] for l in all_leads) / total, 1) if total > 0 else 0
        
        balances = [l['client'].get('balance', 0) for l in all_leads]
        avg_balance = round(sum(balances) / len(balances), 1) if balances else 0
        max_balance = max(balances) if balances else 0
        min_balance = min(balances) if balances else 0
        
        ages = [l['client'].get('age', 0) for l in all_leads if l['client'].get('age', 0) > 0]
        avg_age = round(sum(ages) / len(ages), 1) if ages else 0
        
        durations = [l['client'].get('duration', 0) for l in all_leads]
        avg_duration = round(sum(durations) / len(durations), 1) if durations else 0
        
        context_parts.append(f"""
📊 **OVERALL STATISTICS:**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- **Total Leads:** {total}
- **High Potential:** {high} ({conv_rate}%)
- **Low Priority:** {low}
- **Average Confidence:** {avg_conf}%

💰 **BALANCE:**
- **Average:** ${avg_balance:,.2f}
- **Highest:** ${max_balance:,.2f}
- **Lowest:** ${min_balance:,.2f}

👤 **AGE:**
- **Average Age:** {avg_age} years

📞 **CALL DURATION:**
- **Average Duration:** {avg_duration} seconds
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
        
        context_parts.append("""
📋 **COMPLETE LEAD LIST (Sorted by Display ID):**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
        
        for lead in all_leads[:100]:
            client = lead['client']
            context_parts.append(f"""
**LEAD #{lead['display_id']}**
👤 **Job:** {client.get('job', 'Unknown')} | **Age:** {client.get('age', '?')}
💍 **Marital:** {client.get('marital', 'Unknown')} | **Education:** {client.get('education', 'Unknown')}
💰 **Balance:** ${client.get('balance', 0):,} | **Duration:** {client.get('duration', 0)}s
🏠 **Housing Loan:** {client.get('housing', 'No')} | **Personal Loan:** {client.get('loan', 'No')}
📋 **Previous Outcome:** {client.get('poutcome', 'Unknown')} | **Contacts:** {client.get('previous', 0)}
🎯 **Prediction:** {lead['prediction']} ({lead['probability_yes']}%) | **Confidence:** {lead['confidence']}%
⭐ **Priority:** {lead['priority']}
───────────────────────────────────────────────────────────────
""")
        
        context_parts.append("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ **IMPORTANT:** Leads are referenced by DISPLAY ID (1, 2, 3...)
   - LEAD #1 is the first lead, LEAD #2 is the second, etc.
   - These IDs are ALWAYS consecutive with NO gaps
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
        
        return {
            'context': "\n".join(context_parts),
            'leads': all_leads,
            'stats': {
                'total': total,
                'high': high,
                'low': low,
                'conv_rate': conv_rate,
                'avg_conf': avg_conf,
                'avg_balance': avg_balance,
                'avg_age': avg_age,
                'avg_duration': avg_duration
            },
            'mode': 'complete'
        }
    
    def get_conversation_memory(self, limit=10):
        return self.conversation_history[-limit:]
    
    def add_to_memory(self, question, answer):
        self.conversation_history.append({'role': 'user', 'content': question})
        self.conversation_history.append({'role': 'assistant', 'content': answer})
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def clear_memory(self):
        self.conversation_history = []
        return True
    
    def ask(self, lead_id, user_id, question, mode='auto'):
        if not self.available:
            return {
                'answer': "❌ AI Assistant is not available.\n\nPlease check:\n1. Your GROQ_API_KEY in .env file\n2. Your internet connection\n3. That you have a valid Groq account",
                'status': 'error'
            }
        
        memory = self.get_conversation_memory()
        
        memory_context = ""
        if memory:
            memory_context = """
    📝 **Previous conversation for context:**
    """
            for msg in memory[-6:]:
                role = "User" if msg['role'] == 'user' else "Assistant"
                memory_context += f"\n{role}: {msg['content']}"
            memory_context += """
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
        
        # LEAD-SPECIFIC MODE
        if (mode == 'lead' or (mode == 'auto' and lead_id and lead_id > 0)):
            lead_data = self.get_single_lead_full(lead_id)
            if not lead_data:
                return {'answer': f"❌ Lead #{lead_id} not found.", 'status': 'error'}
            
            complete_context = self.build_complete_context()
            
            prompt = f"""You are LeadScout AI, a sales advisor.

    ⚠️ **IMPORTANT:** You are in LEAD-SPECIFIC mode. Focus ONLY on the selected lead.

    {memory_context}

    📊 **ALL LEADS DATA (for context and comparison):**
    {complete_context['context']}

    🎯 **FOCUS ON THIS SPECIFIC LEAD:**
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    **LEAD #{lead_data['display_id']}**
    👤 **Job:** {lead_data['client'].get('job', 'Unknown')}
    👤 **Age:** {lead_data['client'].get('age', '?')} years
    💍 **Marital Status:** {lead_data['client'].get('marital', 'Unknown')}
    🎓 **Education:** {lead_data['client'].get('education', 'Unknown')}
    💰 **Balance:** ₦{lead_data['client'].get('balance', 0):,}
    🏠 **Housing Loan:** {lead_data['client'].get('housing', 'No')}
    💳 **Personal Loan:** {lead_data['client'].get('loan', 'No')}
    📞 **Call Duration:** {lead_data['client'].get('duration', 0)} seconds
    📈 **Campaign Attempts:** {lead_data['client'].get('campaign', 0)}
    📋 **Previous Outcome:** {lead_data['client'].get('poutcome', 'Unknown')}
    📞 **Previous Contacts:** {lead_data['client'].get('previous', 0)}
    🎯 **Prediction:** {lead_data['prediction']} ({lead_data['probability_yes']}%)
    📊 **Confidence:** {lead_data['confidence']}%
    ⭐ **Priority:** {lead_data['priority']}
    💬 **Recommendation:** {lead_data['message']}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    USER QUESTION: {question}

    📝 **RESPONSE RULES:**
    1. **FOCUS ONLY on the selected lead** - don't generalize
    2. Provide specific answers about this lead only
    3. Use the lead's data to support your answer
    4. If comparing to other leads, mention them briefly but keep focus on this lead
    5. Be natural and conversational
    6. Use **bold** for key points
    7. End with actionable recommendations specific to this lead

    ANSWER:
    """
        
        # GENERAL INSIGHTS MODE
        else:
            complete_context = self.build_complete_context()
            
            if complete_context['mode'] == 'empty':
                return {
                    'answer': "📊 No leads found in your database.\n\nStart making predictions to get insights!",
                    'status': 'success',
                    'mode': 'general'
                }
            
            prompt = f"""You are LeadScout AI, a data analyst with access to ALL lead data.

    ⚠️ **IMPORTANT:** You are in GENERAL INSIGHTS mode. Analyze all leads and provide comprehensive insights.

    {memory_context}

    📊 **COMPLETE DATABASE CONTEXT:**
    {complete_context['context']}

    USER QUESTION: {question}

    📝 **RESPONSE RULES:**
    1. Answer questions about ALL leads
    2. Use the full dataset to support your answers
    3. Reference specific leads by their DISPLAY ID when giving examples
    4. Provide trends, patterns, and insights from the data
    5. Be natural and conversational
    6. Use **bold** for key points
    7. End with actionable recommendations

    ANSWER:
    """
        
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']
            else:
                # Try to get error message from response
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', response.text[:200])
                except:
                    error_msg = response.text[:200]
                answer = f"❌ **API Error ({response.status_code})**\n\n{error_msg}"
                
        except requests.exceptions.ConnectionError:
            answer = "❌ **Connection Error**\n\nCannot connect to Groq API. Please check your internet connection."
        except requests.exceptions.Timeout:
            answer = "❌ **Timeout Error**\n\nThe request took too long. Please try again."
        except Exception as e:
            answer = f"❌ **AI Error**\n\n{str(e)}"
        
        self.add_to_memory(question, answer)
        
        return {
            'answer': answer,
            'lead_id': lead_id if lead_id and lead_id > 0 else None,
            'question': question,
            'mode': 'Lead-Specific' if lead_id and lead_id > 0 else 'General Insights',
            'timestamp': datetime.now().isoformat(),
            'status': 'success' if not answer.startswith('❌') else 'error'
        }
    
    def get_top_leads(self, user_id, limit=5):
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        leads = cursor.execute('''
            SELECT id, display_id, client_data, prediction, probability_yes, priority
            FROM leads
            WHERE prediction = 'Yes'
            ORDER BY probability_yes DESC, confidence DESC
            LIMIT ?
        ''', (limit,)).fetchall()
        
        conn.close()
        
        result = []
        for lead in leads:
            try:
                client = json.loads(lead['client_data'])
            except:
                client = {}
            result.append({
                'id': lead['id'],
                'display_id': lead['display_id'],
                'job': client.get('job', 'Unknown'),
                'age': client.get('age', '?'),
                'balance': client.get('balance', 0),
                'probability': lead['probability_yes'],
                'priority': lead['priority']
            })
        
        return result
    
    def get_lead_summary(self, display_id, user_id):
        lead = self.get_single_lead_full(display_id)
        
        if not lead:
            return f"❌ Lead #{display_id} not found."
        
        client = lead['client']
        
        return f"""📊 **Lead #{lead['display_id']} Summary**

👤 **Client:** {client.get('job', 'Unknown')} · {client.get('age', '?')} years old
🎓 **Education:** {client.get('education', 'Unknown')}
💍 **Status:** {client.get('marital', 'Unknown')}

💰 **Financial:**
- Balance: ${client.get('balance', 0):,}
- Housing Loan: {client.get('housing', 'No')}
- Personal Loan: {client.get('loan', 'No')}

🎯 **Prediction:** {lead['prediction']} ({lead['probability_yes']}% confidence)
📊 **Priority:** {lead['priority']}

📞 **History:**
- Previous Outcome: {client.get('poutcome', 'Unknown')}
- Contacts: {client.get('previous', 0)}
- Campaign Attempts: {client.get('campaign', 0)}"""
    
    def get_recommendations(self, display_id, user_id):
        lead = self.get_single_lead_full(display_id)
        
        if not lead:
            return f"❌ Lead #{display_id} not found."
        
        client = lead['client']
        recommendations = []
        insights = []
        
        # Balance-based recommendations
        balance = client.get('balance', 0)
        if balance > 5000:
            recommendations.append("💰 **High Balance Client**")
            insights.append("- Suggest premium investment products with higher returns")
            insights.append("- Mention wealth management services")
            insights.append("- Offer exclusive VIP packages")
        elif balance > 1000:
            recommendations.append("💰 **Medium Balance Client**")
            insights.append("- Focus on savings and growth products")
            insights.append("- Discuss term deposit benefits")
            insights.append("- Offer moderate-risk investment options")
        else:
            recommendations.append("💰 **Low Balance Client**")
            insights.append("- Start with basic savings accounts")
            insights.append("- Explain low-entry investment options")
            insights.append("- Build trust before upselling")
        
        # Previous outcome recommendations
        poutcome = client.get('poutcome', 'unknown')
        if poutcome == 'success':
            recommendations.append("✅ **Previous Success**")
            insights.append("- Reference their previous subscription in conversation")
            insights.append("- Ask about their satisfaction with the product")
            insights.append("- Offer loyalty bonuses or upgrades")
        elif poutcome == 'failure':
            recommendations.append("⚠️ **Previous Failure**")
            insights.append("- Acknowledge the previous attempt gently")
            insights.append("- Ask what didn't work before")
            insights.append("- Present a different approach or product")
        
        # Job-based recommendations
        job = client.get('job', 'unknown')
        if job in ['management', 'admin.', 'entrepreneur']:
            recommendations.append("👔 **Professional Client**")
            insights.append("- Use formal business language")
            insights.append("- Emphasize investment growth and ROI")
            insights.append("- Mention tax benefits and wealth preservation")
        elif job in ['technician', 'services']:
            recommendations.append("🔧 **Technical/Skilled Client**")
            insights.append("- Use clear, logical explanations")
            insights.append("- Emphasize long-term security")
            insights.append("- Discuss technical aspects of the product")
        elif job in ['retired', 'housemaid']:
            recommendations.append("📅 **Senior/Domestic Client**")
            insights.append("- Emphasize security and stability")
            insights.append("- Use simple, clear language")
            insights.append("- Be patient and build trust")
        
        # Education-based recommendations
        education = client.get('education', 'unknown')
        if education == 'tertiary':
            recommendations.append("🎓 **Highly Educated**")
            insights.append("- Use technical terms confidently")
            insights.append("- Provide data and statistics")
            insights.append("- Discuss complex features in detail")
        elif education == 'secondary':
            recommendations.append("📚 **Good Education Level**")
            insights.append("- Explain benefits clearly and simply")
            insights.append("- Use relatable examples")
            insights.append("- Balance technical and simple language")
        else:
            recommendations.append("📖 **Keep it Simple**")
            insights.append("- Use very simple language")
            insights.append("- Focus on basic benefits")
            insights.append("- Avoid jargon or technical terms")
        
        # Duration-based recommendations
        duration = client.get('duration', 0)
        if duration > 300:
            recommendations.append("📞 **Engaged Client**")
            insights.append("- Client is interested and engaged")
            insights.append("- Build on previous conversation")
            insights.append("- Go deeper into product features")
        elif duration > 150:
            recommendations.append("📞 **Moderately Engaged**")
            insights.append("- Keep the conversation flowing")
            insights.append("- Ask open-ended questions")
            insights.append("- Identify their specific needs")
        else:
            recommendations.append("📞 **Need to Build Rapport**")
            insights.append("- Start with friendly conversation")
            insights.append("- Ask about their financial goals")
            insights.append("- Build trust before selling")
        
        # Age-based recommendations
        age = client.get('age', 0)
        if 35 <= age <= 55:
            recommendations.append("📅 **Prime Age Client**")
            insights.append("- Likely have disposable income")
            insights.append("- Focus on wealth building")
            insights.append("- Mention long-term planning")
        elif age > 55:
            recommendations.append("📅 **Senior Client**")
            insights.append("- Emphasize security and stability")
            insights.append("- Discuss retirement planning")
            insights.append("- Be patient and respectful")
        else:
            recommendations.append("📅 **Young Client**")
            insights.append("- Focus on long-term benefits")
            insights.append("- Start with basic products")
            insights.append("- Build loyalty early")
        
        # Previous contacts recommendations
        previous = client.get('previous', 0)
        if previous > 2:
            recommendations.append("🔄 **Multiple Contacts**")
            insights.append("- Be concise and direct")
            insights.append("- Reference previous interactions")
            insights.append("- Ask if they have questions from before")
        elif previous > 0:
            recommendations.append("🔄 **Some Previous Contact**")
            insights.append("- Reference past conversations")
            insights.append("- Show consistency")
            insights.append("- Build on existing relationship")
        else:
            recommendations.append("🔄 **First Contact**")
            insights.append("- Make a good first impression")
            insights.append("- Introduce yourself clearly")
            insights.append("- Focus on listening")
        
        # Compile final recommendation
        final_recommendation = f"""
🎯 **COMPREHENSIVE RECOMMENDATION FOR LEAD #{lead['display_id']}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 **SUMMARY:**
This is a {client.get('job', 'Unknown')} client aged {client.get('age', '?')} with a balance of ${client.get('balance', 0):,}.
Their previous outcome was {client.get('poutcome', 'Unknown')} and they have had {client.get('previous', 0)} previous contacts.

🎯 **PRIMARY ACTIONS:**
{chr(10).join([f"- {r}" for r in recommendations])}

💡 **KEY INSIGHTS & STRATEGIES:**
{chr(10).join([f"- {i}" for i in insights])}

📊 **SUPPORTING DATA:**
- **Prediction:** {lead['prediction']} ({lead['probability_yes']}% probability)
- **Confidence:** {lead['confidence']}%
- **Priority:** {lead['priority']}
- **Call Duration:** {client.get('duration', 0)} seconds

⚡ **CALL STRATEGY:**
1. Start with a warm greeting and introduce yourself
2. Reference any previous successful interactions
3. Ask about their current financial goals
4. Present the term deposit as a solution
5. Address any concerns or objections
6. Offer to provide more information if needed

💬 **OPENING SCRIPT:**
"Hello {client.get('job', '')}, this is [Your Name] from the bank. I noticed you've {'successfully subscribed before' if poutcome == 'success' else 'shown interest in our products'} and I wanted to reach out about our new term deposit offering with 8.5% interest. Would you have a few minutes to discuss?"

⚠️ **OBJECTIONS TO PREPARE FOR:**
- "I'm not interested" → Acknowledge and ask about their financial goals
- "I don't have time" → Offer to schedule a call at a better time
- "I'm happy with my current bank" → Emphasize the competitive interest rate
- "I don't trust banks" → Build trust by explaining security measures

📈 **EXPECTED OUTCOME:**
Based on the data, this lead has a {lead['probability_yes']}% chance of subscribing.
{lead['priority']} - {lead['message']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        return final_recommendation

def init_ai_database(db_path='leads.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='leads'")
    if not cursor.fetchone():
        cursor.execute('''
            CREATE TABLE leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                display_id INTEGER,
                client_data TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probability_yes REAL NOT NULL,
                probability_no REAL NOT NULL,
                confidence REAL NOT NULL,
                priority TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'single'
            )
        ''')
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_history'")
    if not cursor.fetchone():
        cursor.execute('''
            CREATE TABLE chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id INTEGER,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (lead_id) REFERENCES leads(id) ON DELETE CASCADE
            )
        ''')
    
    conn.commit()
    conn.close()