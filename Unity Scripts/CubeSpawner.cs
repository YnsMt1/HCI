using UnityEngine;
using System.Collections;  // MAKE SURE THIS IS HERE for IEnumerator

public class CubeSpawner : MonoBehaviour
{
    [Header("Cube Prefabs")]
    public GameObject lightCubePrefab;
    public GameObject mediumCubePrefab;
    public GameObject heavyCubePrefab;

    [Header("Spawn Positions")]
    public Transform spawnPosition;
    public Transform targetPosition;

    private GameObject currentCube;
    private string currentCondition;

    void Start()
    {
        Debug.Log("Cube Spawner Ready! Use SpawnCube(condition)");
    }

    // NEW METHOD: Configure cube weight based on condition
    private void ConfigureCubeWeight(GameObject cube, string condition)
    {
        // Extract haptic weight from condition name
        string hapticWeight = "Light"; // default

        if (condition.Contains("MediumHaptic"))
            hapticWeight = "Medium";
        else if (condition.Contains("HeavyHaptic"))
            hapticWeight = "Heavy";

        // Get or add CubeWeightManager
        CubeWeightManager weightManager = cube.GetComponent<CubeWeightManager>();
        if (weightManager == null)
        {
            weightManager = cube.AddComponent<CubeWeightManager>();
        }

        // Set the weight condition
        weightManager.SetWeightCondition(hapticWeight);

        Debug.Log($"Configured cube with condition: {condition} | Haptic weight: {hapticWeight}");
    }

    public void SpawnCube(string condition)
    {
        if (currentCube != null)
        {
            // END THE PREVIOUS TRIAL BEFORE DESTROYING
            DataLogger oldLogger = currentCube.GetComponent<DataLogger>();
            if (oldLogger != null)
            {
                oldLogger.EndTrial();
            }
            Destroy(currentCube);
        }

        currentCondition = condition;
        GameObject cubeToSpawn = null;

        // CHOOSE PREFAB BASED ON CONDITION
        switch (condition)
        {
            case "LightVisual_LightHaptic":
            case "LightVisual_MediumHaptic":
            case "LightVisual_HeavyHaptic":
                cubeToSpawn = lightCubePrefab;
                break;

            case "MediumVisual_LightHaptic":
            case "MediumVisual_MediumHaptic":
            case "MediumVisual_HeavyHaptic":
                cubeToSpawn = mediumCubePrefab;
                break;

            case "HeavyVisual_LightHaptic":
            case "HeavyVisual_MediumHaptic":
            case "HeavyVisual_HeavyHaptic":
                cubeToSpawn = heavyCubePrefab;
                break;

            default:
                Debug.LogError($"Unknown condition: {condition}");
                return;
        }

        if (cubeToSpawn != null)
        {
            // Add a small delay to ensure clean instantiation
            StartCoroutine(SpawnCubeWithDelay(cubeToSpawn, condition));
        }
        else
        {
            Debug.LogError($"No prefab found for condition: {condition}");
        }
    }

    private IEnumerator SpawnCubeWithDelay(GameObject prefab, string condition)
    {
        yield return new WaitForEndOfFrame();

        Vector3 spawnPos = spawnPosition.position;

        // Offset based on cube size (visual cue)
        if (condition.Contains("LightVisual"))
            spawnPos += Vector3.left * 0.5f;
        else if (condition.Contains("HeavyVisual"))
            spawnPos += Vector3.right * 0.5f;

        currentCube = Instantiate(prefab, spawnPos, Quaternion.identity);

        // IMPORTANT: Configure weight BEFORE starting trial
        ConfigureCubeWeight(currentCube, condition);

        // Ensure DataLogger component exists and condition is set
        DataLogger logger = currentCube.GetComponent<DataLogger>();
        if (logger == null)
        {
            logger = currentCube.AddComponent<DataLogger>();
        }

        // Set condition and start trial
        logger.SetCondition(condition);
        logger.StartTrial();

        Debug.Log($"Spawned {condition} cube at position: {currentCube.transform.position}");
    }

    // Manual control for testing
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Alpha1))
            SpawnCube("LightVisual_LightHaptic");
        if (Input.GetKeyDown(KeyCode.Alpha2))
            SpawnCube("MediumVisual_MediumHaptic");
        if (Input.GetKeyDown(KeyCode.Alpha3))
            SpawnCube("HeavyVisual_HeavyHaptic");
    }

    public GameObject GetCurrentCube() { return currentCube; }
}