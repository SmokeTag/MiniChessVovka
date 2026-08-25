#!/bin/bash
# Training health-check script

PROJECT_DIR="/srv/MiniChessVovka"
SCRIPT_PATH="$PROJECT_DIR/src/scheduled_self_play.py"
PID_FILE="$PROJECT_DIR/training.pid"
HEALTH_FILE="$PROJECT_DIR/training.health"
LOG_FILE="$PROJECT_DIR/training_log.txt"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if current time is in AI training window (2 AM - 10 AM UTC)
is_training_time() {
    current_hour=$(date -u +%H | sed 's/^0*//')
    [ -z "$current_hour" ] && current_hour=0
    [ "$current_hour" -ge 2 ] && [ "$current_hour" -lt 10 ]
}

# Function to check if we're in pre-training window (00:00-02:00 UTC, waiting for training)
is_waiting_time() {
    current_hour=$(date -u +%H | sed 's/^0*//')
    [ -z "$current_hour" ] && current_hour=0
    [ "$current_hour" -lt 2 ]
}

# Function to get next availability window
get_next_availability() {
    current_hour=$(date -u +%H | sed 's/^0*//')
    [ -z "$current_hour" ] && current_hour=0
    
    if [ "$current_hour" -lt 2 ]; then
        echo "Training starts in $((2 - current_hour)) hours (at 02:00 UTC)"
    elif [ "$current_hour" -ge 10 ]; then
        echo "Next run in $((24 - current_hour)) hours (at 00:00 UTC tomorrow)"
    else
        echo "Training active until 10:00 UTC ($((10 - current_hour)) hours left)"
    fi
}

echo "================================"
echo "Mini Chess training health check"
echo "================================"
echo ""

# Check process state
process_running=false
process_active=false
activity_seconds=999999

if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE")
    if ps -p $pid > /dev/null 2>&1; then
        process_running=true
        
        # Check activity via health file
        if [ -f "$HEALTH_FILE" ]; then
            last_update=$(cat "$HEALTH_FILE")
            current_time=$(date +%s)
            activity_seconds=$((current_time - last_update))
            
            if [ $activity_seconds -lt 600 ]; then
                process_active=true
            fi
        fi
    fi
else
    # Check for any running scheduled_self_play.py processes
    if pgrep -f "scheduled_self_play.py" > /dev/null 2>&1; then
        process_running=true
    fi
fi

# Determine overall health status
in_training_time=false
in_waiting_time=false
if is_training_time; then
    in_training_time=true
elif is_waiting_time; then
    in_waiting_time=true
fi

status_color=""
status_icon=""
status_text=""

if [ "$in_training_time" = true ]; then
    # DURING training hours (2-10 AM UTC)
    if [ "$process_running" = true ] && [ "$process_active" = true ]; then
        status_color="$GREEN"
        status_icon="✓"
        status_text="HEALTHY - training active (02:00-10:00 UTC)"
    else
        status_color="$RED"
        status_icon="✗"
        if [ "$process_running" = false ]; then
            status_text="PROBLEM - the process should be running at this hour"
        else
            status_text="PROBLEM - process inactive for ${activity_seconds}s (threshold: 10 min)"
        fi
    fi
elif [ "$in_waiting_time" = true ]; then
    # WAITING period (00:00-02:00 UTC)
    if [ "$process_running" = true ]; then
        status_color="$BLUE"
        status_icon="⏳"
        status_text="WAITING - process is waiting for training to start (02:00 UTC)"
    else
        status_color="$RED"
        status_icon="✗"
        status_text="PROBLEM - the process should be running (waiting for 02:00 UTC)"
    fi
else
    # OUTSIDE working hours (10:00-24:00 UTC)
    if [ "$process_running" = false ]; then
        status_color="$GREEN"
        status_icon="✓"
        status_text="HEALTHY - process correctly stopped (outside working hours)"
    else
        status_color="$YELLOW"
        status_icon="!"
        status_text="NOTE - process is still running (will exit shortly)"
    fi
fi

# Display overall health status
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    OVERALL HEALTH STATUS                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "  ${status_color}${status_icon} ${status_text}${NC}"
echo ""

# Time window information
current_time_utc=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
echo "TIME WINDOW:"
echo "  Current Time: $current_time_utc"
echo "  Timer Start:  00:00 UTC (systemd timer fires)"
echo "  Training:     02:00-10:00 UTC (8-hour training window)"
echo ""
if [ "$in_training_time" = true ]; then
    echo -e "  ${GREEN}Training window is active${NC}"
elif [ "$in_waiting_time" = true ]; then
    echo -e "  ${BLUE}Waiting period (process is waiting for 02:00 UTC)${NC}"
else
    echo -e "  ${YELLOW}Outside working hours (timer fires at 00:00 UTC)${NC}"
fi
echo -e "  $(get_next_availability)"
echo ""
echo "================================"
echo ""

# Detailed process information
echo "📊 PROCESS DETAILS:"
if [ ! -f "$PID_FILE" ]; then
    echo -e "  ${YELLOW}⚠ PID file not found${NC}"
    echo "  Training is probably not running"
    echo ""
    
    # Check whether the processes are running directly
    if pgrep -f "scheduled_self_play.py" > /dev/null 2>&1; then
        echo -e "  ${YELLOW}! But scheduled_self_play.py processes were found:${NC}"
        pgrep -f "scheduled_self_play.py" | while read p; do
            ps -p $p -o pid,ppid,cmd,%cpu,%mem,etime=
        done
    fi
else
    pid=$(cat "$PID_FILE")
    echo "  PID from file: $pid"
    
    # Check whether the process is running
    if ps -p $pid > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓ Process is running${NC}"
        echo ""
        ps -p $pid -o pid,ppid,cmd,%cpu,%mem,etime
        
        # Check resource usage
        echo ""
        cpu=$(ps -p $pid -o %cpu= | tr -d ' ')
        mem=$(ps -p $pid -o %mem= | tr -d ' ')
        
        echo -e "  Resource usage:"
        echo -e "    CPU: ${GREEN}$cpu%${NC}"
        echo -e "    MEM: ${GREEN}$mem%${NC}"
        
        # Check the health file
        if [ -f "$HEALTH_FILE" ]; then
            echo ""
            echo "  Last health update: $activity_seconds sec ago"
            
            if [ $activity_seconds -gt 600 ]; then
                echo -e "  ${RED}✗ WARNING: no updates for more than 10 minutes!${NC}"
            elif [ $activity_seconds -gt 300 ]; then
                echo -e "  ${YELLOW}! No updates for more than 5 minutes${NC}"
            else
                echo -e "  ${GREEN}✓ Process is responsive${NC}"
            fi
        else
            echo ""
            echo -e "  ${YELLOW}⚠ Health file not found${NC}"
        fi
    else
        echo -e "  ${RED}✗ Process with PID $pid is not running${NC}"
        
        # Check whether there are other processes
        if pgrep -f "scheduled_self_play.py" > /dev/null 2>&1; then
            echo -e "  ${YELLOW}! But other processes were found:${NC}"
            pgrep -f "scheduled_self_play.py" | while read p; do
                ps -p $p -o pid,ppid,cmd,%cpu,%mem,etime=
            done
        else
            echo -e "  ${RED}✗ No training processes found at all${NC}"
        fi
    fi
fi

echo ""
echo "================================"
echo "Most recent log entries:"
echo "================================"

if [ -f "$LOG_FILE" ]; then
    tail -n 20 "$LOG_FILE"
else
    echo "Log file not found: $LOG_FILE"
fi

echo ""
echo "================================"
echo "Recommendations:"
echo "================================"
echo ""
echo "If training is not running:"
echo "  1. sudo systemctl start minichesstrain.service"
echo ""
echo "If the process is stalled:"
echo "  1. sudo systemctl restart minichesstrain.service"
echo ""
echo "To view the full logs:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To view the systemd logs:"
echo "  sudo journalctl -u minichesstrain.service -f"
echo ""
