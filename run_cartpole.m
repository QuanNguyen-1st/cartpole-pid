model = 'cartpole';

% Initial state order: [position; velocity; upright angle; angular velocity].
% Use metres, seconds, and radians.
cfg = struct();
cfg.M = 1.0;
cfg.m = 0.1;
cfg.ell = 0.5;
cfg.g = 9.81;
cfg.h = 0.01;
cfg.forceLimit = 10;
cfg.trackLimit = 2.4;
cfg.angleLimit = pi/3;
cfg.initial = [0.25; 0; 8*pi/180; 0];
cfg.bias = 0;
duration = 15;
useIntegral = true;
useDerivative = true;

% Pole placement.
a = poly([-1,-1.2,-2,-3,-4]);
leading = cfg.M * cfg.ell;
cfg.ki = leading * a(6) / cfg.g;
cfg.kp = leading * a(5) / cfg.g;
cfg.kd = (leading * a(4) + cfg.ell * cfg.ki) / cfg.g;
cfg.D = leading * a(2) + cfg.ell * cfg.kd;
cfg.K = leading * a(3) + (cfg.M + cfg.m) * cfg.g + cfg.ell * cfg.kp;
if ~useIntegral
    cfg.ki = 0;
end
if ~useDerivative
    cfg.kd = 0;
    cfg.D = 0;
end

% Apply cfg to the model workspace.
load_system(model);
modelWorkspace = get_param(model, 'ModelWorkspace');
assignin(modelWorkspace, 'cfg', cfg);

% Use one RK4 solver.
set_param(model, 'SolverType', 'Fixed-step', 'Solver', 'ode4', ...
    'FixedStep', num2str(cfg.h, 17), ...
    'StopTime', num2str(ceil(duration / cfg.h) * cfg.h, 17), ...
    'ReturnWorkspaceOutputs', 'on');

open_system(model);
