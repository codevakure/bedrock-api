from fastapi import FastAPI, Request, HTTPException, Query, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import boto3
from botocore.config import Config
import uuid
from datetime import datetime
import json
from io import BytesIO
import asyncio
from botocore.exceptions import ClientError
import threading
from uuid import uuid4

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sync_status = {
    'is_syncing': False,
    'last_sync_start': None,
    'last_sync_complete': None,
    'status': 'Never Synced',
    'error_message': None
}

sync_lock = threading.Lock()

# AWS Configuration
region = "us-east-1"
config = Config(
    region_name=region,
    retries=dict(
        max_attempts=3,
        mode='standard'
    )
)

bedrock_agent_runtime_client = boto3.client(
    "bedrock-agent-runtime", 
    config=config,
    region_name=region
)
s3_client = boto3.client("s3", region_name=region)
bucket = "loan-documents-123"

# Model configuration
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
model_arn = f'arn:aws:bedrock:{region}::foundation-model/{model_id}'

# Pydantic models
class QueryRequest(BaseModel):
    prompt: str
    document_id: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    filename: str
    original_filename: str

# Helper functions
def get_retrieve_config(document_id: Optional[str] = None) -> Dict[str, Any]:
    config = {
        'type': 'KNOWLEDGE_BASE',
        'knowledgeBaseConfiguration': {
            'knowledgeBaseId': 'SSEVRJKTXH',
            'modelArn': model_arn,
            'retrievalConfiguration': {
                'vectorSearchConfiguration': {
                    'numberOfResults': 3
                }
            }
        }
    }
    
    if document_id:
        config['knowledgeBaseConfiguration']['retrievalConfiguration']['vectorSearchConfiguration']['filter'] = {
            'stringContains': {
                'key': 'x-amz-bedrock-kb-source-uri',
                'value': document_id
            }
        }
    
    return config

async def stream_generate(prompt: str, document_id: Optional[str] = None):
    try:
        config = get_retrieve_config(document_id)
        response = bedrock_agent_runtime_client.retrieve_and_generate(
            input={'text': prompt},
            retrieveAndGenerateConfiguration=config
        )

        # Process citations and page numbers
        unique_page_numbers = set()
        citations = []

        if "citations" in response:
            for citation in response["citations"]:
                if isinstance(citation, dict) and "retrievedReferences" in citation:
                    for reference in citation["retrievedReferences"]:
                        if isinstance(reference, dict):
                            citation_info = {
                                'document_id': reference.get('location', {}).get('s3Location', {}).get('uri', ''),
                                'page_number': None
                            }
                            if 'metadata' in reference:
                                page_number = reference["metadata"].get("x-amz-bedrock-kb-document-page-number")
                                if page_number is not None:
                                    unique_page_numbers.add(int(page_number))
                                    citation_info['page_number'] = int(page_number)
                            citations.append(citation_info)

        generated_text = response.get('output', {}).get('text', '')
        
        # Stream the response
        chunk_size = 100
        for i in range(0, len(generated_text), chunk_size):
            chunk = generated_text[i:i + chunk_size]
            yield json.dumps({
                "chunk": chunk,
                "is_final": i + chunk_size >= len(generated_text),
                "page_numbers": list(unique_page_numbers) if i + chunk_size >= len(generated_text) else None,
                "citations": citations if i + chunk_size >= len(generated_text) else None
            }) + "\n"
            await asyncio.sleep(0.1)

    except Exception as e:
        yield json.dumps({
            "error": str(e),
            "is_final": True
        }) + "\n"

# API Endpoints
@app.post("/query")
async def query(request: QueryRequest):
    return StreamingResponse(
        stream_generate(request.prompt, request.document_id),
        media_type="text/event-stream"
    )


@app.get("/documents")
async def list_documents(
    request: Request,
    file_type: Optional[str] = Query(None, description="Filter by file type (e.g., 'pdf')")
):
    """List all documents in S3 bucket with optional filtering"""
    try:
        # Get documents from S3
        s3_response = s3_client.list_objects_v2(Bucket=bucket)
        
        documents = []
        if 'Contents' in s3_response:
            for item in s3_response['Contents']:
                key = item['Key']
                file_extension = key.split('.')[-1].lower() if '.' in key else ''
                
                # Get metadata for the file
                try:
                    metadata_response = s3_client.head_object(Bucket=bucket, Key=key)
                    original_filename = metadata_response.get('Metadata', {}).get('original_filename')
                except:
                    original_filename = None
                
                # Determine content type based on extension
                content_type_mapping = {
                    'pdf': 'application/pdf',
                    'doc': 'application/msword',
                    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'txt': 'text/plain'
                }
                content_type = content_type_mapping.get(file_extension, 'application/octet-stream')
                
                # Apply file type filter if specified
                if file_type and file_extension != file_type.lower():
                    continue
                
                doc_info = {
                    'key': key,
                    'size': item['Size'],
                    'last_modified': item['LastModified'].isoformat(),
                    'content_type': content_type,
                    'file_extension': file_extension,
                    'original_filename': original_filename  # Add original filename from metadata
                }
                documents.append(doc_info)

        return {
            'documents': documents,
            'total_count': len(documents)
        }

    except Exception as e:
        error_message = f"Error listing documents: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)


@app.get("/document/{document_key}")
async def get_document(document_key: str):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=document_key)
        content = response['Body'].read()
        
        return Response(
            content=content,
            media_type=response['ContentType'],
            headers={
                'Content-Disposition': f'attachment; filename="{document_key}"'
            }
        )

    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document-details/{document_key}")
async def get_document_details(document_key: str):
    try:
        response = s3_client.head_object(Bucket=bucket, Key=document_key)
        
        return {
            'key': document_key,
            'size': response['ContentLength'],
            'last_modified': response['LastModified'].isoformat(),
            'content_type': response.get('ContentType', 'application/octet-stream'),
            'metadata': response.get('Metadata', {})
        }

    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def update_sync_status(status: str, error_message: Optional[str] = None):
    with sync_lock:
        sync_status['status'] = status
        sync_status['error_message'] = error_message
        
        if status == 'In Progress':
            sync_status['is_syncing'] = True
            sync_status['last_sync_start'] = datetime.now().isoformat()
            sync_status['last_sync_complete'] = 'NA'
        elif status in ['Completed', 'Failed']:
            sync_status['is_syncing'] = False
            sync_status['last_sync_complete'] = datetime.now().isoformat()

# Add Upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Generate unique filename
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        unique_filename = f"{str(uuid4())}.{file_extension}"
        
        # Read file content
        content = await file.read()
        
        # Upload to S3
        s3_client.upload_fileobj(
            BytesIO(content),
            bucket,
            unique_filename,
            ExtraArgs={
                'ContentType': file.content_type,
                'Metadata': {
                    'original_filename': file.filename
                }
            }
        )

        return JSONResponse({
            'message': 'File uploaded successfully',
            'filename': unique_filename,
            'original_filename': file.filename
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add Sync endpoints
def sync_knowledge_base():
    """Synchronize the knowledge base with documents in S3"""
    try:
        update_sync_status('In Progress')
        
        bedrock_agent = boto3.client('bedrock-agent', region_name=region)
        
        # Start ingestion job
        response = bedrock_agent.start_ingestion_job(
            knowledgeBaseId='SSEVRJKTXH',
            jobName=f'sync-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )
        
        # Wait for completion
        waiter = bedrock_agent.get_waiter('ingestion_job_complete')
        waiter.wait(
            knowledgeBaseId='SSEVRJKTXH',
            jobId=response['ingestionJob']['ingestionJobId'],
            WaiterConfig={
                'Delay': 30,
                'MaxAttempts': 60
            }
        )
        
        # Check final status
        job_response = bedrock_agent.get_ingestion_job(
            knowledgeBaseId='SSEVRJKTXH',
            jobId=response['ingestionJob']['ingestionJobId']
        )
        
        if job_response['ingestionJob']['status'] == 'SUCCEEDED':
            update_sync_status('Completed')
        else:
            update_sync_status('Failed', f"Ingestion job failed with status: {job_response['ingestionJob']['status']}")
            
    except Exception as e:
        error_message = f"Sync failed: {str(e)}"
        print(error_message)
        update_sync_status('Failed', error_message)

@app.post("/sync")
async def start_sync():
    """Start knowledge base synchronization"""
    try:
        bedrock_agent = boto3.client('bedrock-agent', region_name=region)
        
        # Get data source
        data_sources = bedrock_agent.list_data_sources(
            knowledgeBaseId='SSEVRJKTXH'
        )
        
        if not data_sources.get('dataSourceSummaries'):
            raise HTTPException(
                status_code=404,
                detail="No data sources found for knowledge base"
            )
            
        data_source_id = data_sources['dataSourceSummaries'][0]['dataSourceId']
        
        # Check for existing in-progress sync
        in_progress_jobs = bedrock_agent.list_ingestion_jobs(
            knowledgeBaseId='SSEVRJKTXH',
            dataSourceId=data_source_id,
            filters=[{
                'attribute': 'STATUS',
                'operator': 'EQ',
                'values': ['IN_PROGRESS']
            }]
        )
        
        if in_progress_jobs.get('ingestionJobSummaries'):
            return JSONResponse({
                'error': 'Sync already in progress',
                'job_id': in_progress_jobs['ingestionJobSummaries'][0]['ingestionJobId'],
                'started_at': in_progress_jobs['ingestionJobSummaries'][0]['startedAt'].isoformat()
            }, status_code=409)
        
        # Start new sync
        new_job = bedrock_agent.start_ingestion_job(
            knowledgeBaseId='SSEVRJKTXH',
            dataSourceId=data_source_id
        )
        
        return JSONResponse({
            'message': 'Sync started successfully',
            'job_id': new_job['ingestionJob']['ingestionJobId'],
            'started_at': new_job['ingestionJob']['startedAt'].isoformat()
        })
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        
        if error_code == 'ThrottlingException':
            return JSONResponse({
                'error': 'Rate limit exceeded. Please try again later.',
                'details': error_message
            }, status_code=429)
        elif error_code == 'ValidationException':
            return JSONResponse({
                'error': 'Invalid request',
                'details': error_message
            }, status_code=400)
        else:
            # Log unexpected AWS errors
            print(f"AWS Error in start_sync: {error_code} - {error_message}")
            raise HTTPException(status_code=500, detail=str(e))
            
    except Exception as e:
        # Log unexpected errors
        print(f"Unexpected error in start_sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/list-jobs")
async def list_all_jobs():
    try:
        bedrock_agent = boto3.client('bedrock-agent', region_name=region)
        response = bedrock_agent.list_ingestion_jobs(
            knowledgeBaseId='SSEVRJKTXH',
            dataSourceId='9M0ZTJMEN1'
        )
        return JSONResponse(response)
    except Exception as e:
        return JSONResponse({"error": str(e)})
    

@app.get("/sync/status")
async def get_sync_status():
    try:
        bedrock_agent = boto3.client('bedrock-agent', region_name=region)
        
        data_sources = bedrock_agent.list_data_sources(
            knowledgeBaseId='SSEVRJKTXH'
        )
        
        if 'dataSourceSummaries' in data_sources and data_sources['dataSourceSummaries']:
            data_source_id = data_sources['dataSourceSummaries'][0]['dataSourceId']
            
            # First check for any in-progress jobs
            in_progress_jobs = bedrock_agent.list_ingestion_jobs(
                knowledgeBaseId='SSEVRJKTXH',
                dataSourceId=data_source_id,
                filters=[{
                    'attribute': 'STATUS',  # Changed from 'status' to 'STATUS'
                    'operator': 'EQ',       # Changed from 'equals' to 'EQ'
                    'values': ['IN_PROGRESS']
                }]
            )
            
            # If there's an in-progress job, return its status
            if 'ingestionJobSummaries' in in_progress_jobs and in_progress_jobs['ingestionJobSummaries']:
                in_progress_job = in_progress_jobs['ingestionJobSummaries'][0]
                # Convert UTC to local timezone
                local_start = in_progress_job['startedAt'].astimezone()
                return JSONResponse({
                    'is_syncing': True,
                    'status': 'In Progress',
                    'last_sync_start': local_start.isoformat(),
                    'last_sync_complete': None,
                    'error_message': None
                })
            
            # If no in-progress job, get the latest completed job
            latest_jobs = bedrock_agent.list_ingestion_jobs(
                knowledgeBaseId='SSEVRJKTXH',
                dataSourceId=data_source_id,
                sortBy={
                    'attribute': 'STARTED_AT',
                    'order': 'DESCENDING'
                },
                maxResults=1
            )
            
            if 'ingestionJobSummaries' in latest_jobs and latest_jobs['ingestionJobSummaries']:
                latest_job = latest_jobs['ingestionJobSummaries'][0]
                
                # Convert UTC to local timezone
                local_start = latest_job['startedAt'].astimezone()
                local_complete = latest_job['updatedAt'].astimezone()
                return JSONResponse({
                    'is_syncing': False,
                    'status': 'Completed' if latest_job['status'] == 'COMPLETE' else latest_job['status'],
                    'last_sync_start': local_start.isoformat(),
                    'last_sync_complete': local_complete.isoformat(),
                    'error_message': None if latest_job['status'] == 'COMPLETE' else f"Sync failed with status: {latest_job['status']}"
                })
            
            # If no jobs found, return default sync status
            return JSONResponse({
                'is_syncing': False,
                'status': 'No sync jobs found',
                'last_sync_start': None,
                'last_sync_complete': None,
                'error_message': None
            })
        
        return JSONResponse({
            'is_syncing': False,
            'status': 'No data sources found',
            'last_sync_start': None,
            'last_sync_complete': None,
            'error_message': None
        })
        
    except Exception as e:
        print(f"Error in get_sync_status: {e}")
        return JSONResponse({
            'is_syncing': False,
            'status': 'Error',
            'last_sync_start': None,
            'last_sync_complete': None,
            'error_message': str(e)
        })
        
        
@app.delete("/document/{document_key}")
async def delete_document(document_key: str):
    try:
        # Delete from S3
        s3_client.delete_object(
            Bucket=bucket,
            Key=document_key
        )
        
        return JSONResponse({
            'message': f'Document {document_key} deleted successfully',
            'deleted_key': document_key
        })
        
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Add health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "service": "bedrock-api",
        "timestamp": datetime.now().isoformat()
    })
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)