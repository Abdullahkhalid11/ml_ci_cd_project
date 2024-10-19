pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("ml-app:${env.BUILD_ID}")
                }
            }
        }
        
        stage('Push to Registry') {
            steps {
                script {
                    docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-credentials') {
                        docker.image("ml-app:${env.BUILD_ID}").push()
                    }
                }
            }
        }
        
        stage('Deploy') {
            steps {
                sh "docker run -d -p 5000:5000 ml-app:${env.BUILD_ID}"
            }
        }
    }
}