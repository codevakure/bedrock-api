import os
import boto3
from flask import Flask, request, jsonify
import pdfplumber
from botocore.client import Config

app = Flask(__name__)

# AWS configuration
region = "us-east-1"
bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name=region)

# Bedrock model configuration
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
model_arn = f'arn:aws:bedrock:{region}::foundation-model/{model_id}'
knowledge_base_id = 'U0VI2VMZV1'  # Set the knowledge base ID here

# Extract text from a PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            if len(text) > 5000:  # Limit to the first 5000 characters
                break
    return text.strip()

# Download a file from S3
def download_file_from_s3(bucket_name, key):
    """Download a file from S3 and store it locally."""
    try:
        # Create a local file path using the file key name (S3 object key)
        local_filename = os.path.join(os.getcwd(), key.replace('/', '_'))  # Replace slashes for Windows compatibility
        boto3.client('s3').download_file(bucket_name, key, local_filename)  # Download the file locally
        return local_filename
    except Exception as e:
        return f"Error downloading file: {str(e)}"

@app.route('/query', methods=['POST'])
def query():
    """Endpoint to query the Bedrock API with a user prompt."""
    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    # Query the Bedrock model
    try:
        response = bedrock_agent_runtime_client.retrieve_and_generate(
            input={
                'text': prompt
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': knowledge_base_id,
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
                },
            }
        )

        generated_text = response['output']['text']
        citations = response.get('citations', [])
        
        # # Prepare citations
        # citation_list = []
        # for citation in citations:
        #     for reference in citation.get('retrievedReferences', []):
        #         citation_list.append(reference.get('location'))

        return jsonify({
            'generated_text': generated_text,
            # 'citations': citation_list
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/combine_prompts', methods=['POST'])
def combine_prompts():
    """Endpoint to query multiple prompts separately and return their summaries."""
    data = request.get_json()
    prompts = data.get('prompts')

    if not prompts or not isinstance(prompts, list):
        return jsonify({'error': 'A list of prompts is required'}), 400

    responses = []

    # Query the Bedrock model for each individual prompt
    for prompt in prompts:
        try:
            response = bedrock_agent_runtime_client.retrieve_and_generate(
                input={
                    'text': prompt
                },
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                        'knowledgeBaseId': knowledge_base_id,
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
                    },
                }
            )

            generated_summary = response['output']['text']
            responses.append({'prompt': prompt, 'summary': generated_summary})

        except Exception as e:
            # Capture any errors for individual prompts
            responses.append({'prompt': prompt, 'error': str(e)})

    return jsonify({'responses': responses}), 200


if __name__ == '__main__':
    app.run(debug=True)
