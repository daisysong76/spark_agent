// Jenkinsfile
pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                // Pull code from Git or GitHub
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh """
                   # Example: create a virtualenv or conda environment
                   # This depends on your Jenkins environment
                   pip install -r data_ingestion/requirements.txt
                   pip install -r embedding_generation/requirements.txt
                   pip install -r ai_agent/requirements.txt
                   # If using PySpark, might also need:
                   # pip install pyspark
                """
            }
        }

        stage('Data Ingestion') {
            steps {
                sh """
                   # Example Spark-submit for data ingestion
                   spark-submit \
                     --master yarn \
                     data_ingestion/spark_ingestion.py \
                     --input s3://my-bucket/raw/ \
                     --output s3://my-bucket/staging-ingested/
                """
            }
        }

        stage('Data Cleaning') {
            steps {
                sh """
                   spark-submit \
                     --master yarn \
                     data_ingestion/spark_cleaning.py \
                     --input s3://my-bucket/staging-ingested/ \
                     --output s3://my-bucket/cleaned/
                """
            }
        }

        stage('Embedding Generation') {
            steps {
                sh """
                   spark-submit \
                     --master yarn \
                     embedding_generation/spark_generate_embeddings.py \
                     --input s3://my-bucket/cleaned/ \
                     --output s3://my-bucket/embeddings/
                """
            }
        }

        stage('Vector DB Updates') {
            steps {
                sh """
                   python vector_store/store_embeddings.py \
                     --embedding_dir s3://my-bucket/embeddings/ \
                     --milvus_host <your_milvus_host> \
                     --milvus_port <your_milvus_port> \
                     --collection_name documents
                """
            }
        }

        stage('Build & Push Docker Image') {
            steps {
                script {
                    // Example Docker build & push to Docker Hub or your private registry
                    sh """
                       docker build -t my-registry/my-ai-agent:latest ai_agent/.
                       docker login -u $DOCKERHUB_USERNAME -p $DOCKERHUB_PASSWORD
                       docker push my-registry/my-ai-agent:latest
                    """
                }
            }
        }

        stage('Deploy to Environment') {
            steps {
                // This could be Kubernetes deployment, Docker Compose, or another approach
                sh """
                   # Example: kubectl apply
                   kubectl apply -f deployment/k8s/deployment.yaml
                   kubectl apply -f deployment/k8s/service.yaml
                """
            }
        }

    } // End of stages

    post {
        always {
            // Archive logs, push notifications, etc.
            echo "Pipeline finished. Check above logs for status."
        }
        success {
            echo "Pipeline completed successfully."
        }
        failure {
            echo "Pipeline failed. Please review the logs."
        }
    }
}
