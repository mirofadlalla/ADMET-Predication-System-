#!/bin/bash
# ADMET Inference Deployment - Bash Script
# Quick setup for containerized deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo "=================================="
    echo " $1"
    echo "=================================="
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}→${NC} $1"
}

# Check prerequisites
check_docker() {
    print_header "Checking Prerequisites"
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found"
        echo "Install Docker from: https://www.docker.com"
        exit 1
    fi
    print_success "Docker found"
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose not found"
        exit 1
    fi
    print_success "Docker Compose found"
}

# Build Docker image
build_image() {
    print_header "Building Docker Image"
    docker build -t admet-inference:latest .
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Start services
start_services() {
    print_header "Starting Services"
    docker-compose up -d
    if [ $? -eq 0 ]; then
        print_success "Services started"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

# Verify services
verify_services() {
    print_header "Verifying Services"
    
    echo "Waiting for API to start (max 30 seconds)..."
    
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            print_success "API is healthy"
            return 0
        fi
        sleep 1
    done
    
    print_error "API did not start within 30 seconds"
    return 1
}

# Test API
test_api() {
    print_header "Testing API"
    
    SMILES="CC(=O)OC1=CC=CC=C1C(=O)O"
    
    RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d "{\"smiles\": \"$SMILES\"}")
    
    if echo "$RESPONSE" | grep -q "predictions"; then
        print_success "API test successful"
        echo "Sample response:"
        echo "$RESPONSE" | jq . || echo "$RESPONSE"
    else
        print_error "API test failed"
        echo "$RESPONSE"
        return 1
    fi
}

# Show usage
show_usage() {
    echo "ADMET Inference Deployment Script"
    echo ""
    echo "Usage: ./deploy.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  check              Check Docker installation"
    echo "  build              Build Docker image"
    echo "  start              Start services"
    echo "  stop               Stop services"
    echo "  logs               Show service logs"
    echo "  test               Test API"
    echo "  setup              Full setup (build + start + verify)"
    echo "  help               Show this help message"
    echo ""
}

# Main script
main() {
    case "${1:-help}" in
        check)
            check_docker
            ;;
        build)
            check_docker
            build_image
            ;;
        start)
            start_services
            ;;
        stop)
            print_header "Stopping Services"
            docker-compose down
            print_success "Services stopped"
            ;;
        logs)
            docker-compose logs -f admet-api
            ;;
        test)
            test_api
            ;;
        setup)
            check_docker
            build_image
            start_services
            verify_services
            test_api
            print_header "Setup Complete"
            echo "Access API at: http://localhost:8000/docs"
            ;;
        *)
            show_usage
            ;;
    esac
}

# Run main function
main "$@"
