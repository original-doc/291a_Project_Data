from typing import Dict
import re


class SourceCodeParser:
    """Parser for the new structured text format"""
    
    @staticmethod
    def parse_text_field(text: str) -> Dict[str, str]:
        """
        Parse the structured text field into components
        
        Text format:
        --- Meta Data ---
        Repo: pytorch-lightning
        Path: src\\lightning\\...
        Function Name: function_name
        ...
        
        --- Docstring ---
        Documentation text...
        
        --- Code ---
        def function_name():
            ...
        """
        sections = {}
        
        # Extract Meta Data section
        meta_match = re.search(r'--- Meta Data ---\n(.*?)(?=\n---|\Z)', text, re.DOTALL)
        if meta_match:
            meta_text = meta_match.group(1)
            sections['meta'] = {}
            
            # Parse meta fields
            sections['meta']['repo'] = SourceCodeParser._extract_field(meta_text, 'Repo')
            sections['meta']['path'] = SourceCodeParser._extract_field(meta_text, 'Path')
            sections['meta']['func_name'] = SourceCodeParser._extract_field(meta_text, 'Function Name')
            sections['meta']['language'] = SourceCodeParser._extract_field(meta_text, 'Language')
            sections['meta']['partition'] = SourceCodeParser._extract_field(meta_text, 'Partition')
        
        # Extract Docstring section
        docstring_match = re.search(r'--- Docstring ---\n(.*?)(?=\n---|\Z)', text, re.DOTALL)
        if docstring_match:
            sections['docstring'] = docstring_match.group(1).strip()
        else:
            sections['docstring'] = ''
        
        # Extract Code section
        code_match = re.search(r'--- Code ---\n(.*?)(?=\n---|\Z)', text, re.DOTALL)
        if code_match:
            sections['code'] = code_match.group(1).strip()
        else:
            sections['code'] = ''
        
        return sections
    
    @staticmethod
    def _extract_field(text: str, field_name: str) -> str:
        """Extract a field value from meta text"""
        match = re.search(rf'^{re.escape(field_name)}:\s*(.+)$', text, re.MULTILINE)
        return match.group(1).strip() if match else ''
    
    @staticmethod
    def parse_document(doc: Dict) -> Dict:
        """Parse and return a simplified record for source code documents.

        Output schema:
        - index: int
        - source_type: 'structured'
        - path: str
        - func_name: str
        - docstring_summary: str
        - text: str (embedding text)
        """
        sections = SourceCodeParser.parse_text_field(doc.get('text', ''))
        func_name = sections.get('meta', {}).get('func_name', '')
        path = sections.get('meta', {}).get('path', '')
        docstring = sections.get('docstring', '')
        code = sections.get('code', '')

        assert len(func_name) > 0, "Function name should not be empty"

        # Create docstring summary (first line)
        if docstring:
            first_line = docstring.split('\n')[0].strip()
            summary = first_line[:256]
        else:
            summary = ''

        text = f"{func_name} {func_name} {summary} {docstring} {code[:500]}".strip()

        return {
            'index': doc.get('index', 0),
            'source_type': 'sourcecode',
            'path': path or doc.get('file') or doc.get('entry_filename', ''),
            'func_name': func_name,
            'docstring_summary': summary,
            'text': text,
        }

class DiscussionParser:
    """Parser for GitHub Discussions-like documents.

    Builds a combined raw_text using title, bodyText, and answer.bodyText:
    "title\n bodyText \n Answer: <answer body>"
    """

    @staticmethod
    def parse_document(doc: Dict) -> Dict:
        """Parse and return a simplified record for discussion documents.

        Output schema:
        - index: int
        - source_type: 'discussion'
        - path: str
        - title: str
        - text: str (embedding text)
        """
        title = doc.get('title', '')
        body = doc.get('bodyText', '')
        # Try common answer shapes: single 'answer' or list 'answers'
        ans = ''
        answer = doc.get('answer') or {}
        if isinstance(answer, dict):
            ans = answer.get('bodyText', '') or answer.get('text', '')
        if not ans:
            answers = doc.get('answers') or []
            if isinstance(answers, list) and answers:
                first = answers[0] if isinstance(answers[0], dict) else {}
                ans = first.get('bodyText', '') or first.get('text', '')

        combined = f"{title}\n {body} \n Answer: {ans}".strip()

        file_name = doc.get('entry_filename') or doc.get('url') or doc.get('file') or title
        idx = doc.get('index', doc.get('id', 0))

        return {
            'index': idx,
            'source_type': 'discussion',
            'path': file_name or '',
            'title': title,
            'text': combined,
        }