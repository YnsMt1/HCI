using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;
using System.Collections;

public class XRWeightController : MonoBehaviour
{
    [Header("Weight Settings")]
    public float mass = 1f;
    public float drag = 0.2f;
    public float angularDrag = 0.3f;

    [Header("Haptic Vibration")]
    public float grabVibration = 0.1f;

    [Header("Physics-Based Movement")]
    public float followStrength = 30f;  // How hard it tries to follow hand
    public float maxVelocity = 5f;      // Speed limit when grabbed

    private Rigidbody rb;
    private UnityEngine.XR.Interaction.Toolkit.Interactables.XRGrabInteractable grabInteractable;
    private Transform attachTransform;
    private bool isGrabbed = false;
    private XRBaseController activeController;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            rb = gameObject.AddComponent<Rigidbody>();
        }

        grabInteractable = GetComponent<UnityEngine.XR.Interaction.Toolkit.Interactables.XRGrabInteractable>();
        if (grabInteractable == null)
        {
            Debug.LogError("No XRGrabInteractable found on " + gameObject.name);
        }
    }

    void Start()
    {
        InitializePhysics();

        if (grabInteractable != null)
        {
            // CRITICAL: Set movement type to velocity tracking
            grabInteractable.movementType = UnityEngine.XR.Interaction.Toolkit.Interactables.XRBaseInteractable.MovementType.VelocityTracking;

            // These settings make weight feel more pronounced
            grabInteractable.throwOnDetach = true;
            grabInteractable.smoothPosition = false;
            grabInteractable.smoothRotation = false;
            grabInteractable.trackPosition = true;
            grabInteractable.trackRotation = true;

            grabInteractable.selectEntered.AddListener(OnGrabbed);
            grabInteractable.selectExited.AddListener(OnReleased);
        }
    }

    public void InitializePhysics()
    {
        if (rb == null)
        {
            rb = GetComponent<Rigidbody>();
            if (rb == null) rb = gameObject.AddComponent<Rigidbody>();
        }

        rb.mass = mass;
        rb.linearDamping = drag;
        rb.angularDamping = angularDrag;
        rb.isKinematic = false;
        rb.useGravity = true;
        rb.interpolation = RigidbodyInterpolation.Interpolate;
        rb.collisionDetectionMode = CollisionDetectionMode.Continuous;
        rb.maxLinearVelocity = maxVelocity;

        Debug.Log($"[{gameObject.name}] Physics Initialized: Mass={mass}, Drag={drag}, MaxVel={maxVelocity}");
    }

    void OnGrabbed(SelectEnterEventArgs args)
    {
        isGrabbed = true;
        attachTransform = args.interactorObject.transform;
        activeController = args.interactorObject.transform.GetComponent<XRBaseController>();

        Debug.Log($"[{gameObject.name}] GRABBED: Mass={mass}, Drag={drag}");

        if (grabInteractable != null)
        {
            // Heavier objects = weaker tracking (more lag)
            float trackingDamping = Mathf.Lerp(30f, 5f, mass / 12f);
            grabInteractable.velocityDamping = trackingDamping;

            // Scale throw velocity by weight
            grabInteractable.throwVelocityScale = Mathf.Max(0.1f, 1f / mass);
            grabInteractable.throwSmoothingDuration = 0.1f * mass;
        }

        StartCoroutine(TriggerHaptic(args.interactorObject));
        StartCoroutine(ContinuousHapticFeedback());
    }

    void OnReleased(SelectExitEventArgs args)
    {
        isGrabbed = false;
        attachTransform = null;
        activeController = null;
        Debug.Log($"[{gameObject.name}] Released");
    }

    IEnumerator TriggerHaptic(UnityEngine.XR.Interaction.Toolkit.Interactors.IXRSelectInteractor interactor)
    {
        var controller = interactor.transform.GetComponent<XRBaseController>();
        if (controller != null)
        {
            controller.SendHapticImpulse(grabVibration, 0.3f);
            yield return new WaitForSeconds(0.1f);

            if (mass > 5f)
            {
                controller.SendHapticImpulse(grabVibration * 0.5f, 0.2f);
            }
        }
    }

    // NEW: Continuous haptic feedback while moving heavy objects
    IEnumerator ContinuousHapticFeedback()
    {
        while (isGrabbed)
        {
            if (activeController != null && rb.linearVelocity.magnitude > 0.3f)
            {
                // Heavier objects = stronger continuous vibration when moving
                float movementVibration = Mathf.Clamp01(grabVibration * rb.linearVelocity.magnitude * 0.2f);
                activeController.SendHapticImpulse(movementVibration, 0.05f);
            }
            yield return new WaitForSeconds(0.1f);
        }
    }

    void FixedUpdate()
    {
        if (!isGrabbed || rb == null) return;

        // Apply additional "inertia resistance" based on mass
        if (rb.linearVelocity.magnitude > 0.1f)
        {
            // Heavier objects resist acceleration/deceleration more
            float inertiaResistance = drag * mass * 0.3f;
            Vector3 resistanceForce = -rb.linearVelocity * inertiaResistance;
            rb.AddForce(resistanceForce, ForceMode.Force);
        }

        // Apply additional rotational damping when grabbed
        if (rb.angularVelocity.magnitude > 0.1f)
        {
            float rotationResistance = angularDrag * mass * 0.2f;
            rb.AddTorque(-rb.angularVelocity * rotationResistance, ForceMode.Force);
        }
    }

    void OnDestroy()
    {
        if (grabInteractable != null)
        {
            grabInteractable.selectEntered.RemoveListener(OnGrabbed);
            grabInteractable.selectExited.RemoveListener(OnReleased);
        }
    }
}