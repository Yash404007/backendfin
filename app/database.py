import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import logging
from typing import List, Dict, Optional, Tuple
import json

logger = logging.getLogger(__name__)

class FinanceDocumentDB:
    """PostgreSQL database handler for finance documents"""
    
    def __init__(self, host: str, database: str, user: str, password: str, port: int = 5432):
        """Initialize database connection"""
        self.connection_params = {
            'host': host,
            'database': database,
            'user': user,
            'password': password,
            'port': port
        }
        self.conn = None
        self.cursor = None
        
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Database connection established successfully")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            # Documents table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS finance_documents (
                    id SERIAL PRIMARY KEY,
                    document_name VARCHAR(255) NOT NULL,
                    file_type VARCHAR(10) NOT NULL,
                    content TEXT NOT NULL,
                    file_size INTEGER,
                    word_count INTEGER,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
            """)
            
            # Chat history table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES finance_documents(id) ON DELETE CASCADE,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Cross-document sessions table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS cross_document_sessions (
                    id SERIAL PRIMARY KEY,
                    session_name VARCHAR(255),
                    document_ids INTEGER[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_name 
                ON finance_documents(document_name)
            """)
            
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_document_id 
                ON chat_history(document_id)
            """)
            
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_timestamp 
                ON chat_history(timestamp)
            """)
            
            self.conn.commit()
            logger.info("Database tables created/verified successfully")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error creating tables: {str(e)}")
            raise
    
    def save_document(self, document_name: str, file_type: str, content: str, 
                     file_size: int, metadata: Optional[Dict] = None) -> int:
        """Save a document to the database"""
        try:
            word_count = len(content.split())
            
            self.cursor.execute("""
                INSERT INTO finance_documents 
                (document_name, file_type, content, file_size, word_count, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (document_name, file_type, content, file_size, word_count, 
                  json.dumps(metadata) if metadata else None))
            
            document_id = self.cursor.fetchone()['id']
            self.conn.commit()
            
            logger.info(f"Document saved successfully with ID: {document_id}")
            return document_id
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error saving document: {str(e)}")
            raise
    
    def get_all_documents(self) -> List[Dict]:
        """Retrieve all documents with metadata"""
        try:
            self.cursor.execute("""
                SELECT id, document_name, file_type, file_size, word_count, 
                       upload_date, last_accessed, 
                       LENGTH(content) as content_length
                FROM finance_documents
                ORDER BY last_accessed DESC
            """)
            
            documents = self.cursor.fetchall()
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def get_document_by_id(self, document_id: int) -> Optional[Dict]:
        """Retrieve a specific document by ID"""
        try:
            self.cursor.execute("""
                SELECT * FROM finance_documents
                WHERE id = %s
            """, (document_id,))
            
            document = self.cursor.fetchone()
            
            # Update last accessed time
            if document:
                self.cursor.execute("""
                    UPDATE finance_documents
                    SET last_accessed = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (document_id,))
                self.conn.commit()
            
            return document
            
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {str(e)}")
            return None
    
    def get_multiple_documents(self, document_ids: List[int]) -> List[Dict]:
        """Retrieve multiple documents by their IDs"""
        try:
            self.cursor.execute("""
                SELECT * FROM finance_documents
                WHERE id = ANY(%s)
                ORDER BY document_name
            """, (document_ids,))
            
            documents = self.cursor.fetchall()
            
            # Update last accessed time for all
            self.cursor.execute("""
                UPDATE finance_documents
                SET last_accessed = CURRENT_TIMESTAMP
                WHERE id = ANY(%s)
            """, (document_ids,))
            self.conn.commit()
            
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving multiple documents: {str(e)}")
            return []
    
    def delete_document(self, document_id: int) -> bool:
        """Delete a document and its chat history"""
        try:
            self.cursor.execute("""
                DELETE FROM finance_documents
                WHERE id = %s
            """, (document_id,))
            
            self.conn.commit()
            logger.info(f"Document {document_id} deleted successfully")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    def save_chat_message(self, document_id: Optional[int], role: str, content: str):
        """Save a chat message"""
        try:
            self.cursor.execute("""
                INSERT INTO chat_history (document_id, role, content)
                VALUES (%s, %s, %s)
            """, (document_id, role, content))
            
            self.conn.commit()
            logger.info(f"Chat message saved for document {document_id}")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error saving chat message: {str(e)}")
    
    def get_chat_history(self, document_id: int, limit: int = 50) -> List[Dict]:
        """Retrieve chat history for a document"""
        try:
            self.cursor.execute("""
                SELECT role, content, timestamp
                FROM chat_history
                WHERE document_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (document_id, limit))
            
            history = self.cursor.fetchall()
            # Reverse to get chronological order
            return list(reversed(history))
            
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            return []
    
    def clear_chat_history(self, document_id: int) -> bool:
        """Clear chat history for a document"""
        try:
            self.cursor.execute("""
                DELETE FROM chat_history
                WHERE document_id = %s
            """, (document_id,))
            
            self.conn.commit()
            logger.info(f"Chat history cleared for document {document_id}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error clearing chat history: {str(e)}")
            return False
    
    def search_documents(self, search_term: str) -> List[Dict]:
        """Search documents by name or content"""
        try:
            self.cursor.execute("""
                SELECT id, document_name, file_type, upload_date, 
                       word_count, LENGTH(content) as content_length
                FROM finance_documents
                WHERE document_name ILIKE %s 
                   OR content ILIKE %s
                ORDER BY upload_date DESC
            """, (f'%{search_term}%', f'%{search_term}%'))
            
            results = self.cursor.fetchall()
            logger.info(f"Found {len(results)} documents matching '{search_term}'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def create_cross_document_session(self, session_name: str, document_ids: List[int]) -> int:
        """Create a cross-document analysis session"""
        try:
            self.cursor.execute("""
                INSERT INTO cross_document_sessions (session_name, document_ids)
                VALUES (%s, %s)
                RETURNING id
            """, (session_name, document_ids))
            
            session_id = self.cursor.fetchone()['id']
            self.conn.commit()
            
            logger.info(f"Cross-document session created with ID: {session_id}")
            return session_id
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error creating cross-document session: {str(e)}")
            raise
    
    def get_cross_document_sessions(self) -> List[Dict]:
        """Retrieve all cross-document sessions"""
        try:
            self.cursor.execute("""
                SELECT s.id, s.session_name, s.document_ids, s.created_at, s.last_used,
                       ARRAY_AGG(d.document_name) as document_names
                FROM cross_document_sessions s
                LEFT JOIN finance_documents d ON d.id = ANY(s.document_ids)
                GROUP BY s.id, s.session_name, s.document_ids, s.created_at, s.last_used
                ORDER BY s.last_used DESC
            """)
            
            sessions = self.cursor.fetchall()
            return sessions
            
        except Exception as e:
            logger.error(f"Error retrieving cross-document sessions: {str(e)}")
            return []
    
    def update_session_last_used(self, session_id: int):
        """Update the last used timestamp for a session"""
        try:
            self.cursor.execute("""
                UPDATE cross_document_sessions
                SET last_used = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (session_id,))
            
            self.conn.commit()
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error updating session last used: {str(e)}")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            stats = {}
            
            # Total documents
            self.cursor.execute("SELECT COUNT(*) as count FROM finance_documents")
            stats['total_documents'] = self.cursor.fetchone()['count']
            
            # Total chat messages
            self.cursor.execute("SELECT COUNT(*) as count FROM chat_history")
            stats['total_messages'] = self.cursor.fetchone()['count']
            
            # Total storage used (approximate)
            self.cursor.execute("""
                SELECT SUM(LENGTH(content)) as total_size 
                FROM finance_documents
            """)
            result = self.cursor.fetchone()
            stats['total_content_size'] = result['total_size'] or 0
            
            # Most recent document
            self.cursor.execute("""
                SELECT document_name, upload_date 
                FROM finance_documents 
                ORDER BY upload_date DESC 
                LIMIT 1
            """)
            recent = self.cursor.fetchone()
            stats['most_recent_document'] = recent['document_name'] if recent else None
            stats['most_recent_date'] = recent['upload_date'] if recent else None
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {}
    
    def check_duplicate_document(self, document_name: str) -> Optional[Dict]:
        """Check if a document with the same name already exists"""
        try:
            self.cursor.execute("""
                SELECT id, document_name, upload_date
                FROM finance_documents
                WHERE document_name = %s
                ORDER BY upload_date DESC
                LIMIT 1
            """, (document_name,))
            
            return self.cursor.fetchone()
            
        except Exception as e:
            logger.error(f"Error checking duplicate document: {str(e)}")
            return None