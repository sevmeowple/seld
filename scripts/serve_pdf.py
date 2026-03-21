#!/usr/bin/env python3
"""Simple HTTP server to serve and display PDF files."""

import http.server
import socketserver
import argparse
from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PDF Viewer</title>
    <style>
        body {{ margin: 0; padding: 0; }}
        iframe {{ width: 100vw; height: 100vh; border: none; }}
    </style>
</head>
<body>
    <iframe src="/{pdf_file}"></iframe>
</body>
</html>
"""

class PDFHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, pdf_file=None, **kwargs):
        self.pdf_file = pdf_file
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = HTML_TEMPLATE.format(pdf_file=self.pdf_file)
            self.wfile.write(html.encode())
        else:
            super().do_GET()

def main():
    parser = argparse.ArgumentParser(description='Serve PDF file via HTTP')
    parser.add_argument('pdf_file', help='Path to PDF file')
    parser.add_argument('--port', type=int, default=8000, help='Port (default: 8000)')
    parser.add_argument('--host', default='0.0.0.0', help='Host (default: 0.0.0.0)')
    args = parser.parse_args()

    pdf_path = Path(args.pdf_file).resolve()
    if not pdf_path.exists():
        print(f"Error: {args.pdf_file} not found")
        return

    import os
    os.chdir(pdf_path.parent)

    handler = lambda *args, **kwargs: PDFHandler(*args, pdf_file=pdf_path.name, **kwargs)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving {args.pdf_file} at http://{args.host}:{args.port}/")
        print(f"Access from browser: http://<server-ip>:{args.port}/")
        httpd.serve_forever()

if __name__ == '__main__':
    main()
