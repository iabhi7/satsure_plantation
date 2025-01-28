#!/bin/bash

# Check if service account exists
if [ -f "/app/service-account.json" ]; then
    echo "Using service account for Earth Engine authentication"
    export GOOGLE_APPLICATION_CREDENTIALS="/app/service-account.json"
else
    echo "No service account found. Please authenticate manually with 'earthengine authenticate'"
    earthengine authenticate --quiet
fi

# Execute the passed command
exec "$@" 