using UnityEngine;

public class CubeWeightManager : MonoBehaviour
{
    [Header("Weight Condition")]
    public string weightCondition = "Light";

    private XRWeightController xrWeightController;

    void Awake()
    {
        xrWeightController = GetComponent<XRWeightController>();
        if (xrWeightController == null)
        {
            Debug.LogError($"[{gameObject.name}] XRWeightController not found!");
        }
    }

    public void SetWeightCondition(string condition)
    {
        weightCondition = condition;
        ApplyWeightCondition();
    }

    public void ApplyWeightCondition()
    {
        if (xrWeightController == null)
        {
            xrWeightController = GetComponent<XRWeightController>();
            if (xrWeightController == null)
            {
                Debug.LogError($"[{gameObject.name}] Cannot apply weight - XRWeightController missing!");
                return;
            }
        }

        // EXTREME VALUES for very noticeable differences during movement
        switch (weightCondition)
        {
            case "Light":
                xrWeightController.mass = 0.3f;              // Super light
                xrWeightController.drag = 0.05f;             // Almost no drag
                xrWeightController.angularDrag = 0.1f;       // Spins freely
                xrWeightController.grabVibration = 0.05f;    // Very subtle
                xrWeightController.maxVelocity = 8f;         // Fast movement limit
                break;

            case "Medium":
                xrWeightController.mass = 4f;                // Medium
                xrWeightController.drag = 4f;                // Moderate drag
                xrWeightController.angularDrag = 2f;         // Some resistance
                xrWeightController.grabVibration = 0.3f;     // Noticeable
                xrWeightController.maxVelocity = 4f;         // Medium speed limit
                break;

            case "Heavy":
                xrWeightController.mass = 15f;               // Very heavy
                xrWeightController.drag = 12f;               // High drag (sluggish!)
                xrWeightController.angularDrag = 6f;         // Hard to rotate
                xrWeightController.grabVibration = 0.9f;     // Strong vibration
                xrWeightController.maxVelocity = 2f;         // Slow movement limit
                break;

            default:
                Debug.LogWarning($"Unknown weight condition: {weightCondition}");
                break;
        }

        xrWeightController.InitializePhysics();

        Debug.Log($"[{gameObject.name}] Applied {weightCondition}: Mass={xrWeightController.mass}, Drag={xrWeightController.drag}, MaxVel={xrWeightController.maxVelocity}");
    }
}