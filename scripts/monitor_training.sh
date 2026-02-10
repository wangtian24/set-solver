#!/bin/bash
# Monitor YOLO training progress

LOG="${1:-weights/detector_training.log}"

echo "=== YOLO Detector Training Monitor ==="
echo "Log: $LOG"
echo ""

# Check if training is running
if pgrep -f "src.train.detector" > /dev/null; then
    echo "🟢 Training is RUNNING"
else
    echo "🔴 Training is NOT running"
fi
echo ""

# Get device info
DEVICE=$(grep "Using device:" "$LOG" 2>/dev/null | head -1)
if [ -n "$DEVICE" ]; then
    echo "$DEVICE"
    echo ""
fi

# Get latest progress line (clean ANSI codes)
echo "=== Latest Progress ==="
tail -50 "$LOG" 2>/dev/null | sed 's/\x1b\[[0-9;]*[mK]//g' | grep -E "^\s+[0-9]+/30" | tail -3

echo ""
echo "=== Validation Metrics ==="
grep -E "(mAP50|mAP\(B\)|precision|recall|box_loss|cls_loss)" "$LOG" 2>/dev/null | sed 's/\x1b\[[0-9;]*[mK]//g' | tail -10

echo ""
echo "=== Time Estimate ==="
# Count completed epochs by looking for validation lines
EPOCHS_DONE=$(grep -c "validating:" "$LOG" 2>/dev/null || echo "0")
if [ "$EPOCHS_DONE" -gt 0 ]; then
    echo "Completed epochs: $EPOCHS_DONE/30"
    REMAINING=$((30 - EPOCHS_DONE))
    echo "Remaining: $REMAINING epochs (est. ~45s/epoch = ~$((REMAINING * 45 / 60)) min)"
else
    echo "No complete epochs yet"
fi
