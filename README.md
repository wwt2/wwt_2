# wwt_2
AHT-UKF
imu_acceleration_ukf = randn(length(true_trajectory_x), 2) * 0.1;

ukf_errors = zeros(length(true_trajectory_x), 1);
ukf_errors_in_zones = [];

 for i = 1:length(true_trajectory_x)

     [U, S, V] = svd(P);
     Sigma = sqrt(S);
     X = [x, x + gamma * (U * Sigma * U')', x - gamma * (U * Sigma * U')'];
     X_pred = zeros(n, 2 * n + 1);
     for j = 1:2 * n + 1
         X_pred(:, j) = state_transition(X(:, j), 1);
     end
     x_pred = X_pred * Wm';
     P_pred = Q;
     for j = 1:2 * n + 1
         P_pred = P_pred + Wc(j) * (X_pred(:, j) - x_pred) * (X_pred(:, j) - x_pred)';
     end
 

     Z = zeros(m, 2 * n + 1);
     for j = 1:2 * n + 1
         Z(:, j) = observation_model(X_pred(:, j));
     end
 
     z_pred = Z * Wm';
     P_zz = R;
     P_xz = zeros(n, m);
     for j = 1:2 * n + 1
         P_zz = P_zz + Wc(j) * (Z(:, j) - z_pred) * (Z(:, j) - z_pred)';
         P_xz = P_xz + Wc(j) * (X_pred(:, j) - x_pred) * (Z(:, j) - z_pred)';
     end
     K = P_xz / P_zz;
 

     y = [estimated_trajectory_x(i); estimated_trajectory_y(i)] - z_pred;
     x = x_pred + K * y;
     P = P_pred - K * P_zz * K';
     x(3) = x(3) + imu_acceleration_ukf(i, 1);
     x(4) = x(4) + imu_acceleration_ukf(i, 2);
 

     filtered_trajectory_x_regular(i) = x(1);
     filtered_trajectory_y_regular(i) = x(2);
 

     %ukf_error = norm([true_trajectory_x(i) - x(1), true_trajectory_y(i) - x(2)]);
     %ukf_errors(i) = ukf_error;
  
     for j = 1:size(corners, 1)
         if filtered_trajectory_x_regular(i) >= corners(j, 1) && filtered_trajectory_x_regular(i) <= corners(j, 1) + 1 &&...
                 filtered_trajectory_y_regular(i) >= corners(j, 2) && filtered_trajectory_y_regular(i) <= corners(j, 2) + 1
      
             error_in_zone = norm([true_trajectory_x(i) - filtered_trajectory_x_regular(i), true_trajectory_y(i) - filtered_trajectory_y_regular(i)]);
             ukf_errors_in_zones(end + 1) = error_in_zone; 
             break;  
         end
     end
 end

if ~isempty(ukf_errors_in_zones)
    ukf_average_error_in_zones = mean(ukf_errors_in_zones);
  
        z = 0.9 * [estimated_trajectory_x(i); estimated_trajectory_y(i)] +...
            0.1 * [x(1); x(2)];


        [U, S, V] = svd(P);
        Sigma = sqrt(S);
        X = [x, x + gamma * (U * Sigma * U')', x - gamma * (U * Sigma * U')'];
        X_pred = zeros(n, 2*n + 1);
        for j = 1:2*n + 1
            X_pred(:, j) = state_transition(X(:, j), 1);
        end
        x_pred = X_pred * Wm';
        P_pred = Q;
        for j = 1:2*n + 1
            P_pred = P_pred + Wc(j) * (X_pred(:, j) - x_pred) * (X_pred(:, j) - x_pred)';
        end


        Z = zeros(m, 2*n + 1);
        for j = 1:2*n + 1
            Z(:, j) = observation_model(X_pred(:, j));
        end

        z_pred = Z * Wm';
        P_zz = R;
        P_xz = zeros(n, m);
        for j = 1:2*n + 1
            P_zz = P_zz + Wc(j) * (Z(:, j) - z_pred) * (Z(:, j) - z_pred)';
            P_xz = P_xz + Wc(j) * (X_pred(:, j) - x_pred) * (Z(:, j) - z_pred)';
        end
        K = P_xz / P_zz;


        residual = z - z_pred; 


        delta_huber_dynamic = delta_huber + (max(current_rss) - nlos_threshold) * 0.05;
