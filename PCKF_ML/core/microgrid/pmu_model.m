function z = pmu_model(x, params)
% PMU observation model -- paper Eqs. (Pe)(Qe).
% z = [Pe; Qe]

delta = x(1);
Pe =  (params.E/params.Xd) * params.V * sin(delta - params.theta);
Qe = -(params.V^2/params.Xd) + (params.E/params.Xd)*params.V*cos(delta - params.theta);
z  = [Pe; Qe];
end
