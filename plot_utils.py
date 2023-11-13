import torch 
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep


def draw_cms_label(ax: plt.Axes, label: str = "Preliminary", rlabel: str = "NPLM", fontsize: int = 28):
    """
    Draw a CMS label on the given matplotlib Axes object.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Axes object to draw the label on.
    label : str, optional
        The label text to display (default is "Preliminary").
    rlabel : str, optional
        The right label text to display (default is "NPLM").
    fontsize : int, optional
        The font size to use for the label (default is 28).
    """
    hep.cms.label(
        ax       = ax,
        data     = True,
        label    = label,
        rlabel   = rlabel,
        fontsize = fontsize
    )
    

def set_label_font(ax: plt.Axes, fontsize: int = 28):
    """
    Set the font size of the x and y axis labels of a matplotlib Axes object.
    
    Parameters:
    ax (matplotlib.pyplot.Axes): The Axes object to modify.
    fontsize (int): The font size to set the labels to. Default is 28.
    """
    ax.set_xlabel(ax.get_xlabel(), fontsize = fontsize)
    ax.set_ylabel(ax.get_ylabel(), fontsize = fontsize)
    
    
def set_tick_font(ax: plt.Axes, fontsize: int = 28):
    """
    Set the font size of the tick labels for the given Axes object.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Axes object for which to set the tick font size.
    fontsize : int, optional
        The font size to use for the tick labels. Default is 28.
    """
    ax.tick_params(axis = "x", labelsize = fontsize, which = "major")
    ax.tick_params(axis = "y", labelsize = fontsize, which = "major")
    
    
def draw_grid(ax: plt.Axes):
    """
    Draw a grid on the given matplotlib Axes object.
    
    Parameters:
    -----------
    ax: plt.Axes
        The matplotlib Axes object on which to draw the grid.
    """
    ax.grid(True, which="major", axis="both", alpha=0.5, color="gray")
    ax.set_axisbelow(True)


def plot_ref_data(
    feature_ref   : torch.Tensor,
    feature_data  : torch.Tensor,
    weights_ref   : torch.Tensor,
    weights_data  : torch.Tensor,
    figsize       : tuple, 
    bins          : np.ndarray, 
    ratio         : bool = True, 
    h_ratio       : list = [3, 1], 
    fontsize      : int  = 36,
    cms           : bool = False,
    cms_label     : str = "",
    cms_rlabel    : str = "",
    color_ref     : str = "lightgray",
    color_data    : str = "black",
    ref_hist_type : str = "stepfilled",
    ref_label     : str = "Reference",
    data_label    : str = "Data",
    ref_alpha     : float = 1.0,
    xlabel        : str = "Feature",
    ylabel        : str = "Events",
    ratio_ylabel  : str = "Ratio",
    hist_yscale   : str = "linear",
    ratio_yscale  : str = "linear",
    show_plot     : bool = True,
    save_plot     : bool = False,
    plot_name     : str = "ref_data",
    plot_path     : str = "./",
    plot_format   : str = "pdf",
    return_fig    : bool = False
    ):
    """
    Plots reference and data histograms with ratio plot.
    
    Parameters:
    -----------
    feature_ref : torch.Tensor
        Reference feature tensor.
    feature_data : torch.Tensor
        Data feature tensor.
    weights_ref : torch.Tensor
        Reference weights tensor.
    weights_data : torch.Tensor
        Data weights tensor.
    figsize : tuple
        Figure size.
    bins : np.ndarray
        Bins for the histograms.
    ratio : bool, optional
        Whether to plot the ratio plot, by default True.
    h_ratio : list, optional
        Height ratio of the subplots, by default [3, 1].
    fontsize : int, optional
        Font size, by default 36.
    cms : bool, optional
        Whether to draw CMS label, by default False.
    cms_label : str, optional
        CMS label, by default "".
    cms_rlabel : str, optional
        CMS right label, by default "".
    color_ref : str, optional
        Reference histogram color, by default "lightgray".
    color_data : str, optional
        Data histogram color, by default "black".
    ref_hist_type : str, optional
        Reference histogram type, by default "stepfilled".
    ref_label : str, optional
        Reference histogram label, by default "Reference".
    data_label : str, optional
        Data histogram label, by default "Data".
    ref_alpha : float, optional
        Reference histogram alpha, by default 1.0.
    xlabel : str, optional
        X-axis label, by default "Feature".
    ylabel : str, optional
        Y-axis label, by default "Events".
    ratio_ylabel : str, optional
        Ratio plot Y-axis label, by default "Ratio".
    hist_yscale : str, optional
        Histogram Y-axis scale, by default "linear".
    ratio_yscale : str, optional
        Ratio plot Y-axis scale, by default "linear".
    show_plot : bool, optional
        Whether to show the plot, by default True.
    save_plot : bool, optional
        Whether to save the plot, by default False.
    plot_name : str, optional
        Plot name, by default "ref_data".
    plot_path : str, optional
        Plot path, by default "./".
    plot_format : str, optional
        Plot format, by default "pdf".
    return_fig : bool, optional
        Whether to return the figure and axes objects, by default False.
    """
    if ratio:
        fig, axes = plt.subplots(
            figsize             = figsize, 
            constrained_layout  = True, 
            nrows               = 2, 
            ncols               = 1, 
            sharex              = True, 
            gridspec_kw         = {"height_ratios": h_ratio}
        )
        ax_hist  = axes[0]
        ax_ratio = axes[1]
    else:
        fig, ax_hist = plt.subplots(
            figsize             = figsize, 
            constrained_layout  = True
        )
        
    ax_hist.set_xlim(bins[0], bins[-1])
    
    bincenters = 0.5*(bins[1:]+bins[:-1])
    
    if cms:
        draw_cms_label(ax=ax_hist, label=cms_label, rlabel=cms_rlabel, fontsize=fontsize)
    set_label_font(ax=ax_hist, fontsize=fontsize)
    set_tick_font(ax=ax_hist, fontsize=fontsize-4)

    
    # histograms
    ref_hist,  _ = np.histogram(feature_ref.flatten(),  bins=bins, weights=weights_ref.flatten())
    data_hist, _ = np.histogram(feature_data.flatten(), bins=bins, weights=weights_data.flatten())
    
    # reference
    ax_hist.hist(bincenters, bins=bins, weights=ref_hist,  histtype=ref_hist_type, label=ref_label, color=color_ref, alpha=ref_alpha)

    # data
    ax_hist.errorbar(
        x          = bincenters, 
        y          = data_hist, 
        yerr       = np.sqrt(data_hist), 
        marker     = "o",
        ls         = "",
        lw         = 2,
        label      = data_label,
        color      = color_data,
        markersize = 10,
        elinewidth = 2,
        capsize    = 5,
        capthick   = 2
    )
    
    ax_hist.legend(fontsize=fontsize-6)
    ax_hist.set_ylabel(ylabel, fontsize=fontsize)
    ax_hist.set_yscale(hist_yscale)
    if not ratio:
        ax_hist.set_xlabel(xlabel, fontsize=fontsize)
    
    if ratio:
        y_ratio = data_hist / ref_hist
        y_err_ratio = np.sqrt(data_hist) / ref_hist
        y_mask = y_ratio != 0
        y_ratio     = y_ratio[y_mask]
        y_err_ratio = y_err_ratio[y_mask]
        bincenters_ratio  = bincenters[y_mask]
    
        ax_ratio.axhline(y=1, ls="--", color="black", lw=2)
        set_label_font(ax=ax_ratio, fontsize=fontsize)
        set_tick_font(ax=ax_ratio, fontsize=fontsize-4)
        
        ax_ratio.errorbar(
            x          = bincenters_ratio, 
            y          = y_ratio,
            yerr       = y_err_ratio,
            marker     = "o",
            ls         = "",
            lw         = 2,
            color      = color_data, 
            markersize = 10,
            elinewidth = 2,
            capsize    = 5,
            capthick   = 2
        )

        ax_ratio.set_xlabel(xlabel, fontsize=fontsize)
        ax_ratio.set_ylabel(ratio_ylabel, fontsize=fontsize)
        
        if ratio_yscale == "log":
            ax_ratio.set_yscale(ratio_yscale)
        elif ratio_yscale == "linear":
            ratio_yabs_max = np.max(np.abs(ax_ratio.get_ylim())) 
            ax_ratio.set_ylim(0, ratio_yabs_max)

    if save_plot:
        fig.savefig(plot_path + plot_name + "." + plot_format, format=plot_format, bbox_inches="tight", dpi=300, facecolor="white")
        
    if show_plot:
        plt.show()
        
    if return_fig:
        return fig, axes
    
    return


def plot_ref_data_reco(
    feature_ref    : torch.Tensor,
    feature_data   : torch.Tensor,
    weights_ref    : torch.Tensor,
    weights_data   : torch.Tensor,
    prediction_ref : torch.Tensor,
    figsize        : tuple, 
    bins           : np.ndarray, 
    ratio          : bool         = True, 
    h_ratio        : list         = [3, 1], 
    fontsize       : int          = 36,
    cms            : bool         = False,
    cms_label      : str          = "",
    cms_rlabel     : str          = "",
    color_ref      : str          = "lightgray",
    color_data     : str          = "black",
    ref_hist_type  : str          = "stepfilled",
    pred_hist_type : str          = "step",
    ref_label      : str          = "Reference",
    data_label     : str          = "Data",
    ref_alpha      : float        = 1.0,
    color_reco     : str          = "red",
    reco_label     : str          = "Reco",
    lw_reco        : float        = 3,
    alpha_reco     : float        = 1.0,
    xlabel         : str          = "Feature",
    ylabel         : str          = "Events",
    ratio_ylabel   : str          = "Ratio",
    hist_yscale    : str          = "linear",
    ratio_yscale   : str          = "linear",
    show_plot      : bool         = True,
    save_plot      : bool         = False,
    plot_name      : str          = "ref_data",
    plot_path      : str          = "./",
    plot_format    : str          = "pdf",
    return_fig     : bool         = False,
    binned_reco    : bool         = False,
    grid_reco      : torch.Tensor = None,
    grid_pred      : torch.Tensor = None,
    ):
    """
    Plots the reference, data and reconstructed histograms for a given feature.
    
    Args:
    - feature_ref: torch.Tensor of shape (n_samples_ref, n_features) containing the reference feature values.
    - feature_data: torch.Tensor of shape (n_samples_data, n_features) containing the data feature values.
    - weights_ref: torch.Tensor of shape (n_samples_ref,) containing the weights for the reference samples.
    - weights_data: torch.Tensor of shape (n_samples_data,) containing the weights for the data samples.
    - prediction_ref: torch.Tensor of shape (n_samples_ref,) containing the reconstructed feature values for the reference samples.
    - figsize: tuple of integers (width, height) in inches specifying the size of the figure.
    - bins: np.ndarray of shape (n_bins+1,) containing the edges of the bins for the histograms.
    - ratio: boolean indicating whether to plot the ratio of data to reference histograms.
    - h_ratio: list of two integers specifying the height ratio between the histogram and ratio plots.
    - fontsize: integer specifying the font size for the labels and ticks.
    - cms: boolean indicating whether to add the CMS label to the plot.
    - cms_label: string specifying the CMS label text.
    - cms_rlabel: string specifying the CMS label text on the right side of the plot.
    - color_ref: string specifying the color for the reference histogram.
    - color_data: string specifying the color for the data histogram.
    - ref_hist_type: string specifying the histogram type for the reference histogram.
    - pred_hist_type: string specifying the histogram type for the reconstructed histogram.
    - ref_label: string specifying the label for the reference histogram.
    - data_label: string specifying the label for the data histogram.
    - ref_alpha: float specifying the alpha value for the reference histogram.
    - color_reco: string specifying the color for the reconstructed histogram.
    - reco_label: string specifying the label for the reconstructed histogram.
    - lw_reco: float specifying the line width for the reconstructed histogram.
    - alpha_reco: float specifying the alpha value for the reconstructed histogram.
    - xlabel: string specifying the label for the x-axis.
    - ylabel: string specifying the label for the y-axis.
    - ratio_ylabel: string specifying the label for the y-axis of the ratio plot.
    - hist_yscale: string specifying the y-axis scale for the histogram plot.
    - ratio_yscale: string specifying the y-axis scale for the ratio plot.
    - show_plot: boolean indicating whether to show the plot.
    - save_plot: boolean indicating whether to save the plot.
    - plot_name: string specifying the name of the plot file.
    - plot_path: string specifying the path to save the plot file.
    - plot_format: string specifying the format of the plot file.
    - return_fig: boolean indicating whether to return the figure and axes objects.
    - binned_reco: boolean indicating whether to plot the reconstructed histogram as binned data.
    - grid_reco: torch.Tensor of shape (n_bins_reco,) containing the bin centers for the reconstructed histogram.
    - grid_pred: torch.Tensor of shape (n_bins_reco,) containing the reconstructed feature values for the bin centers.
    
    Returns:
    - If return_fig is True, returns the figure and axes objects.
    - Otherwise, returns None.
    """
    if ratio:
        fig, axes = plt.subplots(
            figsize             = figsize, 
            constrained_layout  = True, 
            nrows               = 2, 
            ncols               = 1, 
            sharex              = True, 
            gridspec_kw         = {"height_ratios": h_ratio}
        )
        ax_hist  = axes[0]
        ax_ratio = axes[1]
    else:
        fig, ax_hist = plt.subplots(
            figsize             = figsize, 
            constrained_layout  = True
        )
        
    ax_hist.set_xlim(bins[0], bins[-1])
    
    bincenters = 0.5*(bins[1:]+bins[:-1])
    
    if cms:
        draw_cms_label(ax=ax_hist, label=cms_label, rlabel=cms_rlabel, fontsize=fontsize)
    set_label_font(ax=ax_hist, fontsize=fontsize)
    set_tick_font(ax=ax_hist, fontsize=fontsize-4)

    
    # histograms
    ref_hist,  _ = np.histogram(feature_ref.flatten(),  bins=bins, weights=weights_ref.flatten())
    data_hist, _ = np.histogram(feature_data.flatten(), bins=bins, weights=weights_data.flatten())
    pred_hist, _ = np.histogram(feature_ref.flatten(), bins=bins, weights=weights_ref.flatten()*np.exp(prediction_ref.flatten()))
    
    # reference
    ax_hist.hist(bincenters, bins=bins, weights=ref_hist,  histtype=ref_hist_type, label=ref_label, color=color_ref, alpha=ref_alpha)
    
    # reco
    ax_hist.hist(bincenters, bins=bins, weights=pred_hist, histtype=pred_hist_type, label=reco_label, color=color_reco, lw=lw_reco, alpha=alpha_reco)

    # data
    ax_hist.errorbar(
        x          = bincenters, 
        y          = data_hist, 
        yerr       = np.sqrt(data_hist), 
        marker     = "o",
        ls         = "",
        lw         = 2,
        label      = data_label,
        color      = color_data,
        markersize = 10,
        elinewidth = 2,
        capsize    = 5,
        capthick   = 2
    )
    
    ax_hist.legend(fontsize=fontsize-6)
    ax_hist.set_ylabel(ylabel, fontsize=fontsize)
    ax_hist.set_yscale(hist_yscale)
    if not ratio:
        ax_hist.set_xlabel(xlabel, fontsize=fontsize)
    
    if ratio:
        y_ratio = data_hist / ref_hist
        y_err_ratio = np.sqrt(data_hist) / ref_hist
        y_mask = y_ratio != 0
        y_ratio     = y_ratio[y_mask]
        y_err_ratio = y_err_ratio[y_mask]
        bincenters_ratio  = bincenters[y_mask]
        
        if binned_reco:
            y_reco = pred_hist / ref_hist

            ax_ratio.plot(
                bincenters,
                y_reco,
                ls  = "-",
                lw = 6,
                color = color_reco,
                alpha = alpha_reco
            )
            
        elif grid_reco is not None and grid_pred is not None and not binned_reco:
            y_reco = np.exp(grid_pred) 
            
            ax_ratio.plot(
                grid_reco,
                y_reco,
                ls  = "-",
                lw = 6,
                color = color_reco,
                alpha = alpha_reco
            )
            

        ax_ratio.axhline(y=1, ls="--", color="black", lw=2)
        set_label_font(ax=ax_ratio, fontsize=fontsize)
        set_tick_font(ax=ax_ratio, fontsize=fontsize-4)
        
        ax_ratio.errorbar(
            x          = bincenters_ratio, 
            y          = y_ratio,
            yerr       = y_err_ratio,
            marker     = "o",
            ls         = "",
            lw         = 2,
            color      = color_data, 
            markersize = 10,
            elinewidth = 2,
            capsize    = 5,
            capthick   = 2
        )

        ax_ratio.set_xlabel(xlabel, fontsize=fontsize)
        ax_ratio.set_ylabel(ratio_ylabel, fontsize=fontsize)
        
        if ratio_yscale == "log":
            ax_ratio.set_yscale(ratio_yscale)
        elif ratio_yscale == "linear":
            ratio_yabs_max = np.max(np.abs(ax_ratio.get_ylim())) 
            ax_ratio.set_ylim(-2, ratio_yabs_max+3)

    if save_plot:
        fig.savefig(plot_path + plot_name + "." + plot_format, format=plot_format, bbox_inches="tight", dpi=300, facecolor="white")
        
    if show_plot:
        plt.show()
        
    if return_fig:
        return fig, axes
    
    return


def plot_loss_history(
    n_epochs     : np.ndarray,
    loss_history : np.ndarray,
    epochs_init  : int = 350,
    epochs_norm  : int = 1,
    figsize      : tuple = (14, 10),
    fontsize     : int = 36,
    color        : str = "black",
    lw           : int = 4,
    cms          : bool = False,
    cms_label    : str = "",
    cms_rlabel   : str = "",
    grid         : bool = True,
    xlabel       : str = "Epoch",
    ylabel       : str = "Loss",
    show_plot    : bool = True,
    save_plot    : bool = False,
    plot_name    : str = "loss_history",
    plot_path    : str = "./",
    plot_format  : str = "pdf",
    return_fig   : bool = False
    ):
    """
    Plots the loss history of a neural network training process.

    Args:
    - n_epochs (np.ndarray): An array containing the number of epochs.
    - loss_history (np.ndarray): An array containing the loss history.
    - epochs_init (int): The initial epoch to start plotting from. Default is 350.
    - epochs_norm (int): The normalization factor for the x-axis. Default is 1.
    - figsize (tuple): The size of the figure. Default is (14, 10).
    - fontsize (int): The font size of the labels. Default is 36.
    - color (str): The color of the plot. Default is "black".
    - lw (int): The line width of the plot. Default is 4.
    - cms (bool): Whether to draw the CMS label. Default is False.
    - cms_label (str): The CMS label. Default is "".
    - cms_rlabel (str): The CMS right label. Default is "".
    - grid (bool): Whether to draw the grid. Default is True.
    - xlabel (str): The label for the x-axis. Default is "Epoch".
    - ylabel (str): The label for the y-axis. Default is "Loss".
    - show_plot (bool): Whether to show the plot. Default is True.
    - save_plot (bool): Whether to save the plot. Default is False.
    - plot_name (str): The name of the plot. Default is "loss_history".
    - plot_path (str): The path to save the plot. Default is "./".
    - plot_format (str): The format of the plot. Default is "pdf".
    - return_fig (bool): Whether to return the figure and axis objects. Default is False.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object. Only returned if return_fig is True.
    - ax (matplotlib.axes.Axes): The axis object. Only returned if return_fig is True.
    """
    
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    if cms:
        draw_cms_label(ax=ax, label=cms_label, rlabel=cms_rlabel, fontsize=fontsize)
    if grid:
        draw_grid(ax=ax)
    set_label_font(ax=ax, fontsize=fontsize)
    set_tick_font(ax=ax, fontsize=fontsize)

    ax.plot(np.arange(epochs_init, n_epochs) / epochs_norm, loss_history[epochs_init:], lw=lw, color=color)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    if save_plot:
        fig.savefig(plot_path + plot_name + "." + plot_format, format=plot_format, bbox_inches="tight", dpi=300, facecolor="white") 

    if show_plot:
        plt.show()
        
    if return_fig:
        return fig, ax
    