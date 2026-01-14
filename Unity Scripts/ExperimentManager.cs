using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ExperimentManager : MonoBehaviour
{
    [Header("Experiment Settings")]
    public string participantID = "P01";
    public int trialsPerCondition = 4;

    [Header("Cube Spawner Reference")]
    public CubeSpawner cubeSpawner;

    private List<string> allConditions = new List<string>()
    {
        "LightVisual_LightHaptic", "LightVisual_MediumHaptic", "LightVisual_HeavyHaptic",
        "MediumVisual_LightHaptic", "MediumVisual_MediumHaptic", "MediumVisual_HeavyHaptic",
        "HeavyVisual_LightHaptic", "HeavyVisual_MediumHaptic", "HeavyVisual_HeavyHaptic"
    };

    private List<string> trialOrder = new List<string>();
    private int currentTrial = 0;

    void Start()
    {
        GenerateTrialOrder();
        Debug.Log("Experiment Manager Ready! Press SPACE to start next trial");
        Debug.Log($"Total trials: {trialOrder.Count}");
    }

    void GenerateTrialOrder()
    {
        trialOrder.Clear();

        // Create 4 copies of each condition
        for (int i = 0; i < trialsPerCondition; i++)
        {
            // Shuffle conditions for this block
            List<string> shuffled = new List<string>(allConditions);
            ShuffleList(shuffled);
            trialOrder.AddRange(shuffled);
        }
    }

    void ShuffleList(List<string> list)
    {
        for (int i = 0; i < list.Count; i++)
        {
            string temp = list[i];
            int randomIndex = Random.Range(i, list.Count);
            list[i] = list[randomIndex];
            list[randomIndex] = temp;
        }
    }

    public void StartNextTrial()
    {
        if (currentTrial < trialOrder.Count)
        {
            string condition = trialOrder[currentTrial];
            cubeSpawner.SpawnCube(condition);
            Debug.Log($"Trial {currentTrial + 1}/{trialOrder.Count}: {condition}");
            currentTrial++;
        }
        else
        {
            Debug.Log("EXPERIMENT COMPLETE!");
        }
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            StartNextTrial();
        }
    }
}
