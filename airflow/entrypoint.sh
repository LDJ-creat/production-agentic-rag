#!/usr/bin/env bash
set -euo pipefail

# Clean up any existing PID files and processes
echo "Cleaning up any existing Airflow processes..."
if command -v pkill >/dev/null 2>&1; then
    pkill -f "airflow webserver" || true
    pkill -f "airflow scheduler" || true
fi
rm -f /opt/airflow/airflow-webserver.pid
rm -f /opt/airflow/airflow-scheduler.pid

# Wait a moment for processes to fully terminate
sleep 2

# Initialize Airflow database
echo "Initializing Airflow database..."
airflow db init

# Create admin user with admin/admin credentials
echo "Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin || echo "Admin user already exists"

# Start webserver and scheduler
echo "Starting Airflow webserver and scheduler..."
airflow scheduler &
SCHEDULER_PID=$!

airflow webserver --port 8080 &
WEBSERVER_PID=$!

# Keep container alive while both processes are healthy; exit on first failure.
wait -n "$SCHEDULER_PID" "$WEBSERVER_PID"
exit $?