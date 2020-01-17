clear; clc; close all;

filename = 'trajectory.txt';

T = load(filename);
d = size(T,1);
n = size(T,2)/(d+1);

translation = [];

for i = 1:n
    translation = [translation T(:, (d+1)*i)];
end

figure;
hold on;
plot(translation(1,:), translation(2,:), 'LineWidth', 1);
axis equal;