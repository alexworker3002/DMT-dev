import numpy as np, gudhi, morse_3d, time, sys

def main():
    shape = (64, 64, 64)
    rounds = 5
    print("=" * 80)
    print(f" 3D DMT-THE 终极收官验证 | 规模: 64³ | 寿命过滤阈值: 1e-8")
    print("=" * 80)
    print(f"{'分布':<12} | {'轮次'} | {'H1 BotnK'} | {'H2 BotnK'} | {'状态'}")
    print("-" * 80)
    
    for dist in ["Gaussian", "Uniform"]:
        for r in range(rounds):
            rng = np.random.RandomState(42 + r)
            if dist == "Gaussian": tensor = rng.randn(*shape)
            else: tensor = rng.rand(*shape)

            t0 = time.time()
            res = morse_3d.extract_persistence_3d_morse(tensor)
            
            check_pass = True
            errors = []
            for d in [1, 2]:
                o = np.array([(float(res['births'][i]), float(res['deaths'][i])) 
                             for i in range(len(res['births'])) 
                             if res['dims'][i]==d and abs(res['births'][i]-res['deaths'][i]) > 1e-8])
                
                cc = gudhi.CubicalComplex(vertices=(-tensor).flatten(), dimensions=list(shape))
                cc.compute_persistence()
                g = np.array([(-b,-dt) for dim, (b,dt) in cc.persistence() if dim==d and np.isfinite(dt)])
                
                bd = gudhi.bottleneck_distance(g, o) if len(g)>0 or len(o)>0 else 0.0
                errors.append(bd)
                if bd > 1e-6: check_pass = False
            
            st = "✅ PASS" if check_pass else "❌ FAIL"
            print(f"{dist:<12} | R-{r+1}  | {errors[0]:.6f} | {errors[1]:.6f} | {st}")

if __name__ == "__main__":
    main()
