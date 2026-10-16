function [vp,vs,Ksat,Musat,detail] = rock_physics_forward( ...
    modu,con,Pe,Sw,phi,rho,fsoft,alpha,model)
% Calculates dry/saturated elastic parameters and predicts Vp and Vs.
% MATLAB R2019b compatible version.

if numel(con) < 5
    error('con must contain 5 mineral fractions: [Quartz Clay Calcite Feldspar Dolomite].');
end
if size(modu,1) < 8 || size(modu,2) < 2
    error('modu must be at least 8x2.');
end
if alpha <= 0 || alpha >= 1
    vp = NaN; vs = NaN; Ksat = NaN; Musat = NaN;
    detail = empty_detail(alpha,fsoft);
    return;
end
if fsoft < 0 || fsoft > 1
    vp = NaN; vs = NaN; Ksat = NaN; Musat = NaN;
    detail = empty_detail(alpha,fsoft);
    return;
end

phi = max(min(phi,0.999999),0);
Sw = max(min(Sw,1),0);
fsoft = max(min(fsoft,1),0);

Kq    = modu(1,1); Muq   = modu(1,2);
Kcl   = modu(2,1); Mucl  = modu(2,2);
Kca   = modu(3,1); Muca  = modu(3,2);
Kfe   = modu(4,1); Mufe  = modu(4,2);
Kdol  = modu(5,1); Mudol = modu(5,2);
Kw    = modu(7,1);
Kg    = modu(8,1);


Kf = Sw*Kw + (1-Sw)*Kg;
Kf = 1/(Sw/Kw + (1-Sw)/Kg);
pf = 1.0;
Kf = (Sw*Kw^pf + (1-Sw)*Kg^pf)^pf;
Kf = max(Kf,eps);

c = con(1:5);
c(c<0) = 0;
if sum(c) <= 0
    error('The sum of mineral fractions must be greater than zero.');
end
c = c/sum(c);
cs=c(1); cc=c(2); cca=c(3); cfe=c(4); cd=c(5);

Kv = cs*Kq + cc*Kcl + cca*Kca + cfe*Kfe + cd*Kdol;
Kr = 1/(cs/Kq + cc/Kcl + cca/Kca + cfe/Kfe + cd/Kdol);
Uv = cs*Muq + cc*Mucl + cca*Muca + cfe*Mufe + cd*Mudol;
Ur = 1/(cs/Muq + cc/Mucl + cca/Muca + cfe/Mufe + cd/Mudol);
Km = 0.5*(Kv+Kr);
Um = 0.5*(Uv+Ur);

phi_soft = phi*fsoft;
phi_hard = phi*(1-fsoft);

[Tvol,Tdev] = compute_tf_local(Km,Um,alpha);
P = Tvol/3;
Q = Tdev/5;

if isnan(P) || isinf(P) || isnan(Q) || isinf(Q) || P <= 0 || Q <= 0
    vp = NaN; vs = NaN; Ksat = NaN; Musat = NaN;
    detail = empty_detail(alpha,fsoft);
    detail.P = P; detail.Q = Q;
    return;
end

Khard  = Km * max(1-phi_hard,eps)^P;
Muhard = Um * max(1-phi_hard,eps)^Q;

Pe_MPa = Pe/1e6;
kappa = model.A*Pe_MPa + model.B;
kappa = max(kappa,1e-12);

Kdry = Khard/(1 + phi_soft/kappa);
Mudry = Muhard*Kdry/max(Khard,eps);

if phi_soft <= eps || Kf <= eps
    Ksat = Kdry;
else
    denom = phi/Kf + (1-phi)/Khard - Kdry/(Khard^2);
    if abs(denom) < eps
        if denom >= 0
            denom = eps;
        else
            denom = -eps;
        end
    end
    Ksat = Kdry + (1-(Kdry/Khard)^2)/denom;
end
Musat = Mudry;

Ksat = real(Ksat);
Musat = real(Musat);

if isnan(Ksat) || isinf(Ksat) || isnan(Musat) || isinf(Musat) || Ksat<=0 || Musat<=0 || ...
        isnan(rho) || isinf(rho) || rho<=0
    vp = NaN; vs = NaN;
else
    vp = sqrt((Ksat+4*Musat/3)/rho);
    vs = sqrt(Musat/rho);
end

detail.alpha = alpha;
detail.fsoft = fsoft;
detail.Tvol = Tvol;
detail.Tdev = Tdev;
detail.P = P;
detail.Q = Q;
detail.Kf = Kf;
detail.Km = Km;
detail.Um = Um;
detail.phi_hard = phi_hard;
detail.phi_soft = phi_soft;
detail.kappa = kappa;
detail.Khard = Khard;
detail.Muhard = Muhard;
detail.Kdry = Kdry;
detail.Mudry = Mudry;
detail.Ksat = Ksat;
detail.Musat = Musat;
end

function [Tvol,Tdev] = compute_tf_local(Km,Um,al)
A = -1;
B = 0;

al = min(max(al,1e-8),1-1e-8);
R = 3*Um/(3*Km+4*Um);

theta = al.*(acos(al)-al.*sqrt(1-al.^2))/((1-al.^2).^(3/2));
f = al.^2.*(3*theta-2)./(1-al.^2);

F1 = 1+A*(3*(f+theta)/2-R*(3*f/2+5*theta/2-4/3));
F2 = 1+A*(1+3*(f+theta)/2-R*(3*f+5*theta)/2) + ...
    B*(3-4*R)+A*(A+3*B)*(3-4*R)*(f+theta-R*(f-theta+2*theta^2))/2;
F3 = 1+A*(R*(2-theta)+(1+al^2)*f*(R-1)/(al^2))/2;
F4 = 1+A*(3*theta+f-R*(f-theta))/4;
F5 = A*(R*(f+theta-4/3)-f)+B*theta*(3-4*R);
F6 = 1+A*(1+f-R*(theta+f))+B*(1-theta)*(3-4*R);
F7 = 2+A*(9*theta+3*f-R*(5*theta+3*f))/4+B*theta*(3-4*R);
F8 = A*(1-2*R+f*(R-1)/2+theta*(5*R-3)/2)+B*(1-theta)*(3-4*R);
F9 = A*(f*(R-1)-R*theta)+B*theta*(3-4*R);

Tvol = 3*F1/F2;
Tdev = 2/F3 + 1/F4 + (F4*F5+F6*F7-F8*F9)/(F2*F4);
end

function d = empty_detail(alpha,fsoft)
d.alpha = alpha;
d.fsoft = fsoft;
d.Tvol = NaN;
d.Tdev = NaN;
d.P = NaN;
d.Q = NaN;
d.Kf = NaN;
d.Km = NaN;
d.Um = NaN;
d.phi_hard = NaN;
d.phi_soft = NaN;
d.kappa = NaN;
d.Khard = NaN;
d.Muhard = NaN;
d.Kdry = NaN;
d.Mudry = NaN;
d.Ksat = NaN;
d.Musat = NaN;
end
