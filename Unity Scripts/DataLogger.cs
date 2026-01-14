using UnityEngine;
using System.IO;
using System.Collections.Generic;

public class DataLogger : MonoBehaviour
{
    private string filePath;
    private bool trialActive = false;
    private float trialStartTime;
    private Vector3 startPosition;
    private List<Vector3> movementPath = new List<Vector3>();
    private int overshootCount = 0;
    private int targetEntryCount = 0;
    private int correctionCount = 0;
    private Vector3 lastPosition;

    // Experiment conditions
    public string currentCondition = "LightVisual_LightHaptic";
    private string participantID = "P01";
    private bool isInitialized = false;

    void Start()
    {
        InitializeLogger();
    }

    void InitializeLogger()
    {
        filePath = Application.dataPath + $"/FINAL_METRICS_{participantID}.csv";
        // Only write header if file doesn't exist or is empty
        if (!File.Exists(filePath) || new FileInfo(filePath).Length == 0)
        {
            File.WriteAllText(filePath, "Condition,Time,Event,PosX,PosY,PosZ,Overshoots,Corrections,EntryCount\n");
        }
        Debug.Log("Final Metrics Logger Ready!");
        isInitialized = true;
    }

    void Update()
    {
        if (!isInitialized) return;

        if (trialActive && transform.hasChanged)
        {
            // Add current position to movement path FIRST
            movementPath.Add(transform.position);

            // Detect corrections (sharp direction changes) - ONLY if we have enough points
            if (movementPath.Count >= 3)
            {
                Vector3 currentPos = movementPath[movementPath.Count - 1];
                Vector3 previousPos = movementPath[movementPath.Count - 2];
                Vector3 twoStepsBack = movementPath[movementPath.Count - 3];

                Vector3 currentDirection = (currentPos - previousPos).normalized;
                Vector3 previousDirection = (previousPos - twoStepsBack).normalized;

                float directionChange = Vector3.Dot(currentDirection, previousDirection);

                // If direction change is sharp (angle > 45 degrees)
                if (directionChange < 0.7f)
                {
                    correctionCount++;
                }
            }

            // Log movement
            File.AppendAllText(filePath, $"{currentCondition},{Time.time},MOVING,{transform.position.x},{transform.position.y},{transform.position.z},{overshootCount},{correctionCount},{targetEntryCount}\n");

            lastPosition = transform.position;
            transform.hasChanged = false;
        }
    }

    public void StartTrial()
    {
        if (!isInitialized) InitializeLogger();

        trialActive = true;
        trialStartTime = Time.time;
        startPosition = transform.position;
        movementPath.Clear();
        overshootCount = 0;
        correctionCount = 0;
        targetEntryCount = 0;

        // Initialize movement path and lastPosition
        movementPath.Add(startPosition);
        lastPosition = startPosition;  // ADD THIS LINE

        File.AppendAllText(filePath, $"{currentCondition},{Time.time},TRIAL_START,{startPosition.x},{startPosition.y},{startPosition.z},0,0,0\n");
        Debug.Log($"Trial STARTED - Condition: {currentCondition}");
    }

    // ... rest of your DataLogger methods remain the same ...
    public void EndTrial()
    {
        if (!trialActive) return;

        trialActive = false;
        float completionTime = Time.time - trialStartTime;
        float pathEfficiency = CalculatePathEfficiency();
        float totalDistance = CalculateTotalDistance();

        File.AppendAllText(filePath, $"{currentCondition},{Time.time},TRIAL_END,{transform.position.x},{transform.position.y},{transform.position.z},{overshootCount},{correctionCount},{targetEntryCount}\n");
        File.AppendAllText(filePath, $"{currentCondition},{Time.time},RESULTS_Time,{completionTime:F2},0,0,0,0,0\n");
        File.AppendAllText(filePath, $"{currentCondition},{Time.time},RESULTS_Efficiency,{pathEfficiency:F2},0,0,0,0,0\n");
        File.AppendAllText(filePath, $"{currentCondition},{Time.time},RESULTS_Distance,{totalDistance:F2},0,0,0,0,0\n");

        Debug.Log($"Trial COMPLETED: {completionTime:F2}s, Efficiency: {pathEfficiency:F2}, Entries: {targetEntryCount}, Overshoots: {overshootCount}, Corrections: {correctionCount}");
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("TargetZone") && trialActive)
        {
            targetEntryCount++;

            if (targetEntryCount > 1)
            {
                overshootCount++;
                File.AppendAllText(filePath, $"{currentCondition},{Time.time},OVERSHOOT,{transform.position.x},{transform.position.y},{transform.position.z},{overshootCount},{correctionCount},{targetEntryCount}\n");
            }

            File.AppendAllText(filePath, $"{currentCondition},{Time.time},TARGET_ENTER,{transform.position.x},{transform.position.y},{transform.position.z},{overshootCount},{correctionCount},{targetEntryCount}\n");
        }
    }

    void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("TargetZone"))
        {
            File.AppendAllText(filePath, $"{currentCondition},{Time.time},TARGET_EXIT,{transform.position.x},{transform.position.y},{transform.position.z},{overshootCount},{correctionCount},{targetEntryCount}\n");
        }
    }

    float CalculatePathEfficiency()
    {
        if (movementPath.Count < 2) return 0f;
        float straightDist = Vector3.Distance(startPosition, movementPath[movementPath.Count - 1]);
        float actualDist = CalculateTotalDistance();
        return straightDist / Mathf.Max(actualDist, 0.001f);
    }

    float CalculateTotalDistance()
    {
        float distance = 0f;
        for (int i = 1; i < movementPath.Count; i++)
        {
            distance += Vector3.Distance(movementPath[i - 1], movementPath[i]);
        }
        return distance;
    }

    public void SetCondition(string newCondition)
    {
        currentCondition = newCondition;
    }
}