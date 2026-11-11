within ;
package ToasterECU "Smart toaster control unit for the dSimula x SyncTwin demonstrator"

  model ToasterController
    "Toaster ECU: Idle -> Heating -> Release. Exported as toaster_ctrl.fmu and bound to USD with ovfmi."

    // ---- FMU inputs: raw sensor values, read by ovfmi from the SimReady toaster's physics joints ----
    input Real lever_pos(start = -0.0005, unit = "m")
      "Lever joint position (joint_lever_joint_00, state:linear:physics:position; -0.0595 down .. -0.0005 up)";
    input Real dial_angle(start = 135, unit = "deg")
      "Dial joint angle (joint_dial_joint_00, state:angular:physics:position)";
    input Real cancel_pos(start = -0.0002, unit = "m")
      "Cancel button joint position (joint_cancel_button_joint_00; -0.0032 pressed .. -0.0002 released)";
    input Real chamber_temp(start = 20, unit = "degC")
      "Chamber temperature from the thermal FMU (overlay Chamber.toaster:temperature)";

    // ---- FMU outputs: written by ovfmi into USD attributes ----
    output Real lever_target(unit = "m")
      "Lever drive target: held down while latched (joint_lever_joint_00, drive:linear:physics:targetPosition)";
    output Real heater_power(unit = "W")
      "Heater power (overlay Chamber.toaster:heaterPower, read by the thermal FMU)";
    output Real heater_glow
      "Glow light intensity (overlay HeaterGlow.inputs:intensity)";
    output Real time_remaining(unit = "s")
      "Remaining toasting time (overlay Display.toaster:timeRemaining)";
    output Real eject(unit = "1")
      "Eject pulse (1 for t_eject s) when the timer completes; launches the toast (overlay Chamber.toaster:eject)";
    output Boolean latch_engaged "Latch holds the carriage down";
    output Integer browning_level "Decoded dial setting 1..7";

    // ---- Calibration (versioned by dSimula as the calibration data set) ----
    parameter Real P_heater(unit = "W") = 900 "Rated heater power";
    parameter Real t_level[7](each unit = "s") = {60, 80, 100, 120, 145, 170, 200}
      "Toasting time per browning level, cold start";
    parameter Real T_amb(unit = "degC") = 20 "Reference ambient temperature";
    parameter Real k_warm(unit = "1/K") = 0.002
      "Warm-start compensation: time shortened per kelvin above ambient";
    parameter Real T_max(unit = "degC") = 250 "Over-temperature cut-off";
    parameter Real Ts(unit = "s") = 0.05 "ECU task period";
    parameter Real lever_down_below(unit = "m") = -0.05 "Lever switch threshold";
    parameter Real lever_low(unit = "m") = -0.0595 "Lever travel lower limit (latched)";
    parameter Real lever_high(unit = "m") = -0.0005 "Lever travel upper limit (released)";
    parameter Real cancel_below(unit = "m") = -0.0025 "Cancel switch threshold";
    parameter Real dial_band(unit = "deg") = 45 "Dial angle per browning level";
    parameter Real glow_max = 30000 "Glow light intensity at full heater power";
    parameter Real t_eject(unit = "s") = 0.3 "Length of the eject pulse";

  protected
    Boolean lever_down = lever_pos < lever_down_below "Lever switch";
    Boolean cancel = cancel_pos < cancel_below "Cancel switch";
    Integer state(start = 0, fixed = true) "0 = Idle, 1 = Heating, 2 = Release";
    discrete Real t_end(start = 0, fixed = true) "Planned end of heating";
    Boolean over_temp_trip(start = false, fixed = true)
      "Last cycle ended by the over-temperature cut-off";
    discrete Real eject_until(start = -1, fixed = true) "Eject pulse ends at this time";

  algorithm
    when sample(0, Ts) then
      if pre(state) == 0 then
        // Idle: a lever press starts a cycle (not while cancel is held)
        if lever_down and not cancel then
          state := 1;
          over_temp_trip := false;
          t_end := time + toastTime(browning_level, chamber_temp, t_level, T_amb, k_warm);
        end if;
      elseif pre(state) == 1 then
        // Heating: stop on over-temperature, cancel or timer
        if chamber_temp >= T_max then
          over_temp_trip := true;
          state := 2;
        elseif cancel then
          state := 2;
        elseif time >= t_end then
          // done: release the latch and fire the eject spring
          state := 2;
          eject_until := time + t_eject;
        end if;
      else
        // Release: latch is open; wait until the lever has come up
        if not lever_down then
          state := 0;
        end if;
      end if;
    end when;

  equation
    browning_level = dialLevel(dial_angle, dial_band);
    latch_engaged = state == 1;
    lever_target = if latch_engaged then lever_low else lever_high;
    heater_power = if state == 1 then P_heater else 0;
    heater_glow = glow_max * heater_power / P_heater;
    time_remaining = if state == 1 then max(t_end - time, 0) else 0;
    eject = if time < eject_until then 1 else 0;

    annotation (
      experiment(StartTime = 0, StopTime = 300, Interval = 0.05),
      Documentation(info = "<html>
<p>Digital successor of the analog A0201F timer board (SEVERIN AT 2287 teardown).
Inputs are raw sensor values so that ovfmi can map them directly from the joint states of the
NVIDIA SimReady toaster appliance_toaster_a01; switch thresholds and dial decoding are ECU calibration.</p>
<ul>
<li>Idle &rarr; Heating: lever down and cancel not pressed.</li>
<li>Heating &rarr; Release: chamber_temp &ge; T_max, cancel pressed, or timer elapsed
(only the timer fires the eject pulse that launches the toast).</li>
<li>Release &rarr; Idle: lever is up again.</li>
</ul>
</html>"));
  end ToasterController;

  function dialLevel "Browning level 1..7 from the dial angle (unlimited revolute joint)"
    input Real angle(unit = "deg");
    input Real band(unit = "deg");
    output Integer level;
  protected
    Real a;
  algorithm
    a := angle - 360 * floor(angle / 360);
    level := max(1, min(7, 1 + integer(a / band)));
  end dialLevel;

  function toastTime "Toasting time for a dial level, shortened when the chamber is still warm"
    input Integer level;
    input Real T(unit = "degC");
    input Real t_level[7](each unit = "s");
    input Real T_amb(unit = "degC");
    input Real k_warm(unit = "1/K");
    output Real t(unit = "s");
  algorithm
    t := t_level[max(1, min(7, level))] * max(0.5, 1 - k_warm * max(T - T_amb, 0));
  end toastTime;

  model ToasterThermal
    "Thermal and browning model of the toaster chamber and the toast. Exported as toaster_thermal.fmu (owned by the simulation team)."
    input Real heater_power(start = 0, unit = "W")
      "From overlay Chamber.toaster:heaterPower";
    input Real lever_pos(start = -0.0005, unit = "m")
      "Carriage position = lever joint (joint_lever_joint_00); the toast only browns while it is down";
    output Real chamber_temp(unit = "degC") "To overlay Chamber.toaster:temperature";
    output Real browning "To overlay Chamber.toaster:browning: 0 untoasted, ~1 golden, >1.5 burnt";
    output Real toast_r "Toast colour, red   (Toast material inputs:diffuseColor[0])";
    output Real toast_g "Toast colour, green (Toast material inputs:diffuseColor[1])";
    output Real toast_b "Toast colour, blue  (Toast material inputs:diffuseColor[2])";

    parameter Real C(unit = "J/K") = 350 "Heat capacity of chamber and elements";
    parameter Real G(unit = "W/K") = 3.5 "Heat loss to ambient";
    parameter Real T_amb(unit = "degC") = 20;
    parameter Real T_brown(unit = "degC") = 150 "Browning starts above this temperature";
    parameter Real k_brown(unit = "1/(K.s)") = 8e-4 "About 1 (golden) after a cold level-4 cycle";
    parameter Real lever_down_below(unit = "m") = -0.05 "Carriage counts as down below this position";
    parameter Real out_of_slot_factor = 0.05 "Browning rate while the toast is popped up";

  protected
    Real T(start = 20, fixed = true, unit = "degC");
    Real b(start = 0, fixed = true);
    Real exposure = if lever_pos < lever_down_below then 1 else out_of_slot_factor;

  equation
    C * der(T) = heater_power - G * (T - T_amb);
    der(b) = exposure * k_brown * max(T - T_brown, 0);
    chamber_temp = T;
    browning = b;
    toast_r = toastColor(b, 1);
    toast_g = toastColor(b, 2);
    toast_b = toastColor(b, 3);
  end ToasterThermal;

  function toastColor "Toast colour channel for a browning index: bread -> golden -> dark -> burnt"
    input Real b "browning index";
    input Integer channel "1 = red, 2 = green, 3 = blue";
    output Real c;
  protected
    constant Real knots[4] = {0, 1, 2, 3};
    constant Real colors[4, 3] = [0.93, 0.85, 0.68;   // fresh bread
                                  0.80, 0.55, 0.27;   // golden
                                  0.36, 0.20, 0.09;   // dark
                                  0.10, 0.07, 0.05];  // burnt
    Real x;
    Integer i;
  algorithm
    x := min(max(b, 0), 3);
    i := min(integer(x) + 1, 3);
    c := colors[i, channel] + (x - knots[i]) * (colors[i + 1, channel] - colors[i, channel]);
  end toastColor;


  model ToastFlight
    "Toast carriage and ejection: rides on the lever in the slot, flies a ballistic arc on the eject pulse,
     lands flat on the worktop. Deterministic, so a maturity state replays the same flight. Exported as
     toast_flight.fmu (owned by the simulation team). Positions are in the toaster frame (z = 0 is the worktop)."
    input Real lever_pos(start = -0.0005, unit = "m") "Lever joint position (carriage)";
    input Real eject(start = 0) "Eject pulse from the ECU (overlay Chamber.toaster:eject)";
    output Real toast_x(unit = "m") "Toast centre, toaster frame -> Toast xformOp:translate[0]";
    output Real toast_y(unit = "m") "-> Toast xformOp:translate[1]";
    output Real toast_z(unit = "m") "-> Toast xformOp:translate[2]";
    output Real toast_rot(unit = "deg") "Tumble about the slot axis -> Toast xformOp:rotateXYZ[0]";
    output Integer toast_mode "0 = in slot, 1 = flying, 2 = landed";

    parameter Real slot_y(unit = "m") = -0.026 "Slot centre in the toaster frame";
    parameter Real z_up(unit = "m") = 0.17 "Toast centre height with the lever up";
    parameter Real lever_up(unit = "m") = -0.0005 "Lever position when up";
    parameter Real v_launch[3](each unit = "m/s") = {0, 0.8, 3.0} "Launch velocity (x, y, z)";
    parameter Real turns = 1.25 "Tumble until landing; x.25 lands it flat";
    parameter Real z_rest(unit = "m") = 0.006 "Centre height lying flat on the worktop (half thickness)";
    parameter Real g(unit = "m/s2") = 9.81;
    parameter Real lever_down_below(unit = "m") = -0.05 "Lever push re-inserts a landed toast";
    parameter Real Ts(unit = "s") = 0.002 "Event check period";

  protected
    Integer mode(start = 0, fixed = true);
    discrete Real t0(start = 0, fixed = true) "Launch time";
    discrete Real z0(start = 0.17, fixed = true) "Launch height";
    discrete Real t_fly(start = 1, fixed = true) "Flight time to the worktop";
    discrete Real land_x(start = 0, fixed = true);
    discrete Real land_y(start = 0, fixed = true);
    Real tau = time - t0;
    Real z_carriage = z_up + (lever_pos - lever_up);

  algorithm
    when sample(0, Ts) then
      if pre(mode) == 0 then
        if eject > 0.5 then
          mode := 1;
          t0 := time;
          z0 := z_carriage;
          // time until the centre is back down at z_rest: z0 + vz t - g t^2 / 2 = z_rest
          t_fly := (v_launch[3] + sqrt(v_launch[3] ^ 2 + 2 * g * (z0 - z_rest))) / g;
        end if;
      elseif pre(mode) == 1 then
        if time - t0 >= t_fly then
          mode := 2;
          land_x := v_launch[1] * t_fly;
          land_y := slot_y + v_launch[2] * t_fly;
        end if;
      else
        if lever_pos < lever_down_below then
          mode := 0;  // the user puts the toast back and pushes the lever
        end if;
      end if;
    end when;

  equation
    toast_mode = mode;
    toast_x = if mode == 1 then v_launch[1] * tau elseif mode == 2 then land_x else 0;
    toast_y = if mode == 1 then slot_y + v_launch[2] * tau elseif mode == 2 then land_y else slot_y;
    toast_z = if mode == 1 then max(z0 + v_launch[3] * tau - g * tau ^ 2 / 2, z_rest)
              elseif mode == 2 then z_rest else z_carriage;
    toast_rot = if mode == 1 then 360 * turns * min(tau / t_fly, 1) elseif mode == 2 then 90 else 0;
  end ToastFlight;

  package Tests

    model ToasterCycle
      "Three cycles: cold start at level 4, warm restart at level 7, cancelled cycle"
      ToasterController ctrl;
      ToasterThermal thermal;
      ToastFlight flight;

      // The user pushes the lever for 0.5 s; afterwards the latch drive holds it down.
      Boolean push = (time >= 1 and time < 1.5)
                  or (time >= 150 and time < 150.5)
                  or (time >= 400 and time < 400.5);
    equation
      ctrl.lever_pos = if push or pre(ctrl.latch_engaged) then -0.0595 else -0.0005;
      ctrl.dial_angle = if time < 140 then 150 else 290;  // level 4, then level 7
      ctrl.cancel_pos = if time >= 420 and time < 420.5 then -0.0032 else -0.0002;
      ctrl.chamber_temp = thermal.chamber_temp;
      thermal.heater_power = ctrl.heater_power;
      thermal.lever_pos = ctrl.lever_pos;
      flight.lever_pos = ctrl.lever_pos;
      flight.eject = ctrl.eject;
      annotation (experiment(StartTime = 0, StopTime = 600, Interval = 0.1));
    end ToasterCycle;
  end Tests;

  annotation (version = "1.4.0");
end ToasterECU;
