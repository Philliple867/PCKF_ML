function p = mg_params(cfg, Pm_eff, V, theta)
% Assembles microgrid parameter struct.
% Pm_eff = T_m - P_w(v_t) -- effective mechanical power, updated per timestep.

p.H      = cfg.H;
p.D      = cfg.D;
p.Xd     = cfg.Xd;
p.E      = cfg.E;
p.ws     = cfg.ws;
p.dt     = cfg.dt;
p.V      = cfg.V;
p.theta  = cfg.theta;

if nargin < 2 || isempty(Pm_eff)
    p.Pm_eff = cfg.Pm_eff_nom;
else
    p.Pm_eff = Pm_eff;
end
if nargin >= 3 && ~isempty(V);     p.V     = V;     end
if nargin >= 4 && ~isempty(theta); p.theta = theta; end
end
