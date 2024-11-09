import os
import boto3
from flask import Flask, request, jsonify, Response
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the Bedrock client
region = "us-east-1"
bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name=region)
s3_client = boto3.client("s3", region_name=region)  # Initialize S3 client
bucket = "txcknowledgebasebedrock"  # Replace with your actual S3 bucket name

model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
model_arn = f'arn:aws:bedrock:{region}::foundation-model/{model_id}'

# Configuration for Bedrock API
retrieveGenConfig = {
    'type': 'KNOWLEDGE_BASE',
    'knowledgeBaseConfiguration': {
        'knowledgeBaseId': 'VCJBGHVNSF',
        'modelArn': model_arn,
        "orchestrationConfiguration": {
            "queryTransformationConfiguration": {
                "type": "QUERY_DECOMPOSITION"
            }
        },
        "generationConfiguration": {
            "inferenceConfig": {
                "textInferenceConfig": {
                    "maxTokens": 2048,
                    "temperature": 0.2,
                    "topP": 0.9
                }
            }
        },
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {
                "numberOfResults": 5,
                "overrideSearchType": "HYBRID"
            }
        },
    }
}

def process_prompt(prompt):
    """Helper function to call the Bedrock API and extract unique page numbers."""
    # Call the Bedrock API
    response = bedrock_agent_runtime_client.retrieve_and_generate(
        input={'text': prompt},
        retrieveAndGenerateConfiguration=retrieveGenConfig
    )

    # Extract unique page numbers from citations
    unique_page_numbers = set()  # Using a set to avoid duplicates

    # Check if citations exist and iterate over them
    if "citations" in response:
        for citation in response["citations"]:
            print(citation)
            if isinstance(citation, dict) and "retrievedReferences" in citation:
                for reference in citation["retrievedReferences"]:
                    if isinstance(reference, dict) and "metadata" in reference:
                        print("Metadata fields:", reference.get("metadata", {}))
                        page_number = reference["metadata"].get("x-amz-bedrock-kb-document-page-number")
                        if page_number is not None:
                            unique_page_numbers.add(int(page_number))

    unique_page_numbers = list(unique_page_numbers)
    generated_text = response.get('output', {}).get('text', '')  # Handle cases where text might not be present

    return generated_text, unique_page_numbers

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    prompt = data.get('prompt')
    generated_text, unique_page_numbers = process_prompt(prompt)

    return jsonify({
        "prompt": prompt,
        "generated_text": generated_text,
        "page_numbers": unique_page_numbers
    })

@app.route('/combine-query', methods=['POST'])
def combine_prompts():
    data = request.json
    prompts = data.get('prompts', [])
    results = []

    for prompt in prompts:
        generated_text, unique_page_numbers = process_prompt(prompt)
        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "page_numbers": unique_page_numbers
        })

    return jsonify({
        "results": results
    })

@app.route('/download-pdf', methods=['GET'])
def download_pdf():
    """Endpoint to download a PDF document from S3."""
    bucket_name = bucket
    file_key = request.args.get('file_key')

    if not bucket_name or not file_key:
        return jsonify({'error': 'Bucket name and file key are required'}), 400

    try:
        # Get the PDF file object from S3
        s3_response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        
        # Ensure the content type is correct
        content_type = s3_response['ContentType'] or 'application/pdf'  # Fallback to 'application/pdf'

        # Send the file as a response
        response = Response(
            s3_response['Body'].read(),  # Read the body of the response
            content_type=content_type,
            headers={
                'Content-Disposition': f'attachment; filename="{os.path.basename(file_key)}"'  # Correctly format filename
            }
        )
        
        return response

    except s3_client.exceptions.NoSuchKey:
        return jsonify({'error': 'File not found in S3'}), 404
    except Exception as e:
        # Log the error for debugging
        print(f"Error fetching file from S3: {str(e)}")
        return jsonify({'error': f'Error fetching file from S3: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
