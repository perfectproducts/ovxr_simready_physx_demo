# dbeyond smart toaster (copied for the VR demo)

Copied from the dSimula × SyncTwin demonstrator (`C:\_STASH\DBEYOND\demonstrator\dbeyond`,
registry `synctwin://projects/dbeyond`), with the same layout so the relative paths between layers and
FMUs still resolve. Only what the VR demo needs is here: no counter, no robot, no results.

```
Assets/Products/Toaster_A01/    NVIDIA SimReady appliance_toaster_a01 (simready/, unchanged)
                                + toaster_ecu_overlay.usda + toaster_fmi_binding.usda (ovfmi binding)
Assets/Products/Toast_Slice/    procedural toast slice
FMUs/ToasterECU/                toaster_ctrl.fmu     OpenModelica ECU (source/ToasterECU.mo)
FMUs/ToasterThermal/            toaster_thermal.fmu  chamber thermal, browning, toast colour
FMUs/ToastFlight/               toast_flight.fmu     carriage and ballistic ejection
Scenarios/StudierstubeToaster/  ToasterStation.usda  VR-only: toaster + toast + ToastFlight wiring,
                                slot glow for Minimal rendering, hand-operable part tuning
```

The FMUs are FMI 2.0 co-simulation, win64 binaries only. `vrsim/toaster.py` reads the binding and
runs them with FMPy. To rebuild them, run `omc export_fmu.mos` in the dbeyond repo's `demonstrator/ecu`
(OpenModelica 1.26) and copy the three `.fmu` files here.

`ToasterStation.usda` changes physics only for VR, on top of the product: heavier lever, dial and
button, a preloaded lever spring, and light damping, so per-frame velocity writes from the
controllers can move them. It also switches the PhysX variant to convex colliders. The product
layers stay as published.

## Licenses

- Toaster: NVIDIA SimReady assets, `appliance_toaster_a01`, **CC-BY-4.0**
  (https://huggingface.co/datasets/nvidia/simready-assets, revision
  19512d4dd4b7c0868530efd3ec62277fa2dc1cde).
- ECU overlay, FMI binding, toast slice, FMUs and their Modelica source: dbeyond / SyncTwin demonstrator.
