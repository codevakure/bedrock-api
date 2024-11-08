import os
import json
import boto3
from flask import Flask, request, jsonify
from botocore.client import Config

app = Flask(__name__)

# AWS configuration
region = "us-east-1"
bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name=region)

# Bedrock model configuration
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
model_arn = f'arn:aws:bedrock:{region}::foundation-model/{model_id}'
knowledge_base_id = 'U0VI2VMZV1'  # Set the knowledge base ID here

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
        
        # Prepare citations
        citation_list = []
        for citation in citations:
            for reference in citation.get('retrievedReferences', []):
                citation_list.append(reference.get('location'))

        return jsonify({
            'generated_text': generated_text,
            'citations': citation_list
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
