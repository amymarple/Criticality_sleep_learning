function [bin_centers, probdens] = loglogDist(temp, color, marker, marker_size, dashed, dashed2, mm, MM, shift, plotOrNot)

if size(temp, 1) > size(temp, 2)
    temp = temp';
end


bins = logspace(log10(min(temp)), log10(max(temp)), 10*round(log10(max(temp)/min(temp))));
counts = histc(temp, bins);
counts = counts(1:(end-1));
probdens = counts./diff(bins)/length(temp);
bin_centers = bins(1:(end-1)) + diff(bins)/2;

bin_centers = bin_centers(counts > 0);
probdens = probdens(counts > 0);


if plotOrNot
    if dashed
        lh = plot(bin_centers, shift*probdens, 'Color', color, 'LineWidth', 1.33, 'LineStyle', '--');
        lh.Color = [lh.Color 0.8];
    elseif dashed2
        plot(bin_centers, shift*probdens, 'Color', color, 'LineWidth', 1.5) %, 'LineStyle', '--');
    else
        hold on;
        plot(bin_centers(mm<=bin_centers & bin_centers<MM), shift*probdens(mm<=bin_centers & bin_centers<MM), 'Color', color, 'LineStyle', 'none', 'Marker', marker, 'MarkerSize', marker_size, 'MarkerFaceColor', color)
        plot(bin_centers(bin_centers<mm | bin_centers>=MM), shift*probdens(bin_centers<mm | bin_centers>=MM), 'Color', color, 'LineStyle', 'none', 'Marker', marker, 'MarkerSize', marker_size, 'MarkerFaceColor', 'none')
    end
    
    set(gca, 'xscale', 'log', 'yscale', 'log')
end

end