from graphviz import Digraph

def create_bpmn_flow():
    dot = Digraph(comment="Saudi Hospital BPMN Flow", format="png")
    dot.attr(rankdir='TB', size='10')

    # Emergency Flow
    with dot.subgraph(name="cluster_emergency") as e:
        e.attr(label="🚨 Emergency/Trauma Case", style="filled", color="lightcoral")
        e.node("E_start", "Start (Accident/Red Crescent)", shape="circle")
        e.node("E1", "Triage Nurse Assessment", shape="box")
        e.node("E2", "Quick Registration (ID/Unknown)", shape="box")
        e.node("E3", "Resuscitation & Stabilization (ABCDE)", shape="box")
        e.node("E4", "Diagnostics (Lab, Radiology, ECG)", shape="box")
        e.node("E_decision", "Decision Point", shape="diamond")
        e.node("E5", "Discharge with Instructions", shape="box")
        e.node("E6", "Admit to ICU/Ward", shape="box")
        e.node("E7", "Immediate Surgery (OT)", shape="box")
        e.node("E_end", "End", shape="circle")

        e.edges([("E_start","E1"),("E1","E2"),("E2","E3"),("E3","E4"),("E4","E_decision")])
        e.edge("E_decision","E5", label="Minor")
        e.edge("E_decision","E6", label="Major")
        e.edge("E_decision","E7", label="Life-threatening")
        e.edge("E5","E_end")
        e.edge("E6","E_end")
        e.edge("E7","E_end")

    # Outpatient Flow
    with dot.subgraph(name="cluster_outpatient") as o:
        o.attr(label="🤒 Outpatient (Walk-in/Scheduled)", style="filled", color="lightblue")
        o.node("O_start", "Start (Appointment via Mawid/Private App)", shape="circle")
        o.node("O1", "Registration (ID, Insurance, MRN)", shape="box")
        o.node("O2", "Nurse Vitals & Initial Assessment", shape="box")
        o.node("O3", "Doctor Consultation", shape="box")
        o.node("O_decision", "Decision Point", shape="diamond")
        o.node("O4", "Prescription Only", shape="box")
        o.node("O5", "Diagnostics (Lab/Radiology)", shape="box")
        o.node("O6", "Referral/Specialist", shape="box")
        o.node("O7", "Admit to Hospital", shape="box")
        o.node("O8", "Check-Out & Billing", shape="box")
        o.node("O9", "Pharmacy (Meds Dispensed)", shape="box")
        o.node("O_end", "End", shape="circle")

        o.edges([("O_start","O1"),("O1","O2"),("O2","O3"),("O3","O_decision")])
        o.edge("O_decision","O4", label="Meds")
        o.edge("O_decision","O5", label="Tests")
        o.edge("O_decision","O6", label="Referral")
        o.edge("O_decision","O7", label="Admit")
        o.edge("O4","O8")
        o.edge("O5","O3", label="Return with Results")
        o.edge("O6","O8")
        o.edge("O7","O8")
        o.edge("O8","O9")
        o.edge("O9","O_end")

    # Planned/Chronic Inpatient Flow
    with dot.subgraph(name="cluster_inpatient") as c:
        c.attr(label="🩺 Planned/Chronic Inpatient", style="filled", color="lightgreen")
        c.node("C_start", "Start (Doctor Decision)", shape="circle")
        c.node("C1", "Pre-Admission Testing (Labs, ECG, Insurance)", shape="box")
        c.node("C2", "Admission Day (Wristband, Bed Assigned)", shape="box")
        c.node("C3", "Ward Stay (Rounds, Nursing Care, Dietician, PT)", shape="box")
        c.node("C4", "Procedure (OR, PACU, ICU if needed)", shape="box")
        c.node("C5", "Discharge Planning (Case Manager, Instructions)", shape="box")
        c.node("C6", "Final Billing & Insurance Claim", shape="box")
        c.node("C_end", "End", shape="circle")

        c.edges([("C_start","C1"),("C1","C2"),("C2","C3"),("C3","C4"),("C4","C5"),("C5","C6"),("C6","C_end")])

    return dot

# Generate BPMN flow
bpmn_flow = create_bpmn_flow()
bpmn_flow.render('Hospital Flows', format='svg', view=False)