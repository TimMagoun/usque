%%
clear;
load data;

%%
figure();
plot_q_gt(t, ukfmemcorrectinit, gtmem, "UKF Performance, Mem Grade");
%%
figure();
plot_q_gt(t, ekfmemcorrectinit, gtmem, "EKF Performance, Mem Grade");
%%
figure();
plot_q_gt(t, compmemcorrectinit, gtmem, "Complementary Filter, Mem Grade");

%%
figure();
plot_q_gt(t, naivememcorrectinit, gtmem, "Naive Integration, Mem Grade");

%% Plot gt omega and bias
plot(t, memgtomega);
title("Ground Truth Angular Velocity, Mem Grade");
xlabel("Time (s)");
ylabel("\omega (rad/s)");
improvePlot();
exportgraphics(gcf, "figs/Ground Truth Angular Velocity.png");
%%
plot(t, memnoisyomega);
title("Noisy Angular Velocity, Mem Grade");
xlabel("Time (s)");
ylabel("\omega (rad/s)");
improvePlot();
exportgraphics(gcf, "figs/Noisy Angular Velocity.png");
%%
plot(t, quat2eul(gtmem));
title("Ground Truth Attitude");
legend(["Yaw", "Pitch", "Roll"])
xlabel("Time (s)");
ylabel("\theta (rad)");
improvePlot();
exportgraphics(gcf, "figs/GT Attitude.png");
%%
plot(t, memgtbias);
title("Ground Truth Bias, Mem Grade");
xlabel("Time (s)");
ylabel("\omega_b (rad/s)");
improvePlot();
exportgraphics(gcf, "figs/Ground Truth Bias.png");
%%
plot(t, tacgtbias);
title("Ground Truth Bias, Tac Grade");
xlabel("Time (s)");
ylabel("\omega_b (rad/s)");
improvePlot();
exportgraphics(gcf, "figs/Ground Truth Bias Tac.png");

%%
plot_q_err(t, ukfmemcorrectinit, gtmem, "UKF All Errors, Mem Grade", true);
%%
plot_q_err(t, ukfmemcorrectinit, gtmem, "UKF Error, Mem Grade", false);
plot_q_err(t, ekfmemcorrectinit, gtmem, "EKF Error, Mem Grade", false);
plot_q_err(t, compmemcorrectinit, gtmem, "Complementary Filter Error, Mem Grade", false);
plot_q_err(t, madgmemcorrectinit, gtmem, "Madgwick Filter Error, Mem Grade", false);
plot_q_err(t, naivememcorrectinit, gtmem, "Naive Filter Error, Mem Grade", false);
%%
plot_q_err(t, ukfmemincorrectinit, gtmem, "UKF Error, Mem Grade, Bad Init", false);
plot_q_err(t, ekfmemincorrectinit, gtmem, "EKF Error, Mem Grade, Bad Init", false);
plot_q_err(t, compmemincorrectinit, gtmem, "Complementary Filter Error, Mem Grade, Bad Init", false);
plot_q_err(t, madgmemincorrectinit, gtmem, "Madgwick Filter Error, Mem Grade, Bad Init", false);
plot_q_err(t, naivememincorrectinit, gtmem, "Naive Filter Error, Mem Grade, Bad Init", false);
%%
plot_q_err(t, ekftacincorrectinit, gttac, "EKF Error, Tac Grade, Bad Init", false);
plot_q_err(t, ukftacincorrectinit, gttac, "UKF Error, Tac Grade, Bad Init", false);
plot_q_err(t, comptacincorrectinit, gttac, "Complementary Error, Tac Grade, Bad Init", false);
%%
avg_err(compmemcorrectinit, gtmem)
avg_err(compmemincorrectinit, gtmem)
avg_err(comptaccorrectinit, gttac)
avg_err(comptacincorrectinit, gttac)
avg_err(ekfmemcorrectinit, gtmem)
avg_err(ekfmemincorrectinit, gtmem)
avg_err(ekftaccorrectinit, gttac)
avg_err(ekftacincorrectinit, gttac)
avg_err(madgmemcorrectinit, gtmem)
avg_err(madgmemincorrectinit, gtmem)
avg_err(madgtaccorrectinit, gttac)
avg_err(madgtacincorrectinit, gttac)
avg_err(naivememcorrectinit, gtmem)
avg_err(naivememincorrectinit, gtmem)
avg_err(naivetaccorrectinit, gttac)
avg_err(naivetacincorrectinit, gttac)
avg_err(ukfmemcorrectinit, gtmem)
avg_err(ukfmemincorrectinit, gtmem)
avg_err(ukftaccorrectinit, gttac)
avg_err(ukftacincorrectinit, gttac)

%%
function err = avg_err(q_est, q_gt)
    ypr_err = quat2eul(q_est, "ZYX") - quat2eul(q_gt, "ZYX");
    err = mean(sqrt(vecnorm(ypr_err(:, 2:3), 2, 2)));
end

%%
function plot_q_gt(t, q, gt, title_str)
    hold on;
    plot(t, quat2eul(gt, "ZYX"));
    plot(t, quat2eul(q, "ZYX"), "--");
    legend(["Yaw gt", "Pitch gt", "Roll gt", "Yaw est", "Pitch est", "Roll est"]);
    xlabel("Time (s)");
    ylabel("Angle (rad)");
    title(title_str);
    improvePlot();
    hold off;
    exportgraphics(gcf, "figs/" + title_str + ".png");
end
%%
function plot_q_err(t, q, gt, title_str, plot_yaw)
    figure();
    ypr_err = quat2eul(q, "ZYX") - quat2eul(gt, "ZYX");
    ypr_err = wrapTo2Pi(ypr_err + pi) - pi;
    if plot_yaw
        plot(t, ypr_err);
        legend(["Yaw Err", "Pitch Err", "Roll Err"]);
    else
        plot(t, ypr_err(:, 2:3));
        legend(["Pitch Err", "Roll Err"]);
    end
    xlabel("Time (s)");
    ylabel("Angle (rad)");
    title(title_str);
    improvePlot();
    exportgraphics(gcf, "figs/" + title_str + ".png");
    close gcf;
end

