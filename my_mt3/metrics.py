def onset_f1(pred_pm, ref_pm, tol_ms=50):
    def onsets(pm):
        return sorted(int(n.start*1000) for inst in pm.instruments for n in inst.notes)
    P,R = onsets(pred_pm), onsets(ref_pm)
    i=j=tp=0; used=set()
    while i<len(P) and j<len(R):
        if abs(P[i]-R[j])<=tol_ms: tp+=1; i+=1; j+=1
        elif P[i] < R[j]: i+=1
        else: j+=1
    prec = tp/max(1,len(P)); rec = tp/max(1,len(R))
    f1 = 2*prec*rec/max(1e-9,prec+rec)
    return prec, rec, f1