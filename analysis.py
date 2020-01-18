import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.patheffects as pe

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def plot_progress_synth(losses, y_lims=[(0,0.05),(0,15),(0,0.13)], savename=None):
    
    fig = plt.figure(figsize=(18,9))

    gs = gridspec.GridSpec(2, 2)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    ax1.set_title('Reconstruction Losses', fontsize=30)
    ax2.set_title('Regression Losses', fontsize=30)
    ax3.set_title('Jacobian Losses', fontsize=30)

    ax1.plot(losses['batch_iters'], losses['x_synth'],
             label=r'$x=Dec(y)$')
    ax1.set_ylabel('Loss',fontsize=25)
    ax1.set_ylim(*y_lims[0])

    ax2.plot(losses['batch_iters'], losses['y_synth'],
             label=r'$y=Enc(x)$')
    ax2.plot(losses['batch_iters'], losses['z_synth'],
             label=r'$z=Enc(x)$')
    ax2.set_ylabel('Loss',fontsize=25)
    ax2.set_ylim(*y_lims[1])
    
    ax3.plot(losses['batch_iters'], losses['dxdy_synth'],
             label=r'$\frac{\partial \mathcal{X}_{synth\rightarrow synth}}{\partial y_i}$')
    ax3.set_ylim(*y_lims[2])

    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xlabel('Batch Iterations',fontsize=25)
        ax.tick_params(labelsize=20)
        ax.legend(fontsize=22, ncol=2)
        ax.grid(True)

    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename)
        
    plt.show()

def run_tsne(data_a, data_b, perplex):
    from tsne import bh_sne

    m = len(data_a)

    # Combine data
    t_data = np.row_stack((data_a,data_b))

    # Convert data to float64 matrix. float64 is need for bh_sne
    t_data = np.asarray(t_data).astype('float64')
    t_data = t_data.reshape((t_data.shape[0], -1))

    # Run t-SNE
    vis_data = bh_sne(t_data, perplexity=perplex)
    
    # Separate 2D into x and y axes information
    vis_x_a = vis_data[:m, 0]
    vis_y_a = vis_data[:m, 1]
    vis_x_b = vis_data[m:, 0]
    vis_y_b = vis_data[m:, 1]
    
    return vis_x_a, vis_y_a, vis_x_b, vis_y_b
    
def plot_compare_estimates_resid_obs(x_data, y_data, snr, savename=None, 
                           x_lab=r'$ASPCAP \ \ \ DR14$', 
                           y_lab=r'$(SN\ Cycle$-$GAN) - DR14$',
                           snr_max=200, cmap='Blues', snr_cutoff=100,
                           resid_lims = [[-1000., 1000.], [-2, 2], [-1, 1], 
                             [-1., 1.], [-1., 1.], [-1., 1.]]):
    plt.rcParams['axes.facecolor']='white'
    sns.set_style("ticks")
    plt.rcParams['axes.grid']=True
    plt.rcParams['grid.color']='gray'
    plt.rcParams['grid.alpha']='0.4'

    # label names
    label_names = [r'$T_{\mathrm{eff}}$',r'$\log(g)$',r'$v_{micro}$',r'$[C/H]$',
                   r'$[N/H]$',r'$[O/H]$',r'$[Na/H]$',r'$[Mg/H]$',r'$[Al/H]$',
                   r'$[Si/H]$',r'$[P/H]$',r'$[S/H]$',r'$[K/H]$',r'$[Ca/H]$',
                   r'$[Ti/H]$',r'$[V/H]$',r'[$Cr/H$]',r'$[Mn/H]$',r'$[Fe/H]$',
                   r'$[Co/H]$',r'$[Ni/H]$',r'[$Cu/H$]',r'$[Ge/H]$',r'$[C12/C13]$',r'$v_{macro}$']

    # overplot high s/n
    order = (snr).reshape(snr.shape[0],).argsort()
    x_data = x_data[order]
    y_data = y_data[order]
    snr = snr[order]
    
    # Set maximum S/N
    snr[snr>snr_max]=snr_max

    # Calculate residual, median residual, and std of residuals
    resid = y_data - x_data
    bias = np.median(resid, axis=0)
    scatter = np.std(resid, axis=0)
    resid_a = resid[np.where(snr>=snr_cutoff)[0],:]
    resid_b = resid[np.where(snr<snr_cutoff)[0],:]
    
    # Plot data
    fig = plt.figure(figsize=(40, 90)) 
    gs = gridspec.GridSpec(13, 5,  width_ratios=[4., 1., 1.8, 4., 1.])
    x_plt_indx = 0
    for i in range(y_data.shape[1]):

        # Set column index
        if i>=(int((y_data.shape[1]/2)+1)):
            x_plt_indx=3
        ax0 = plt.subplot(gs[i%int((y_data.shape[1]/2)+1),x_plt_indx])
        #ax0 = axes[i%3,x_plt_indx]
        
        # Plot resid vs x coloured with snr
        points = ax0.scatter(x_data[:,i], resid[:,i], c=snr, s=20, cmap=cmap)

        # Set axes labels
        ax0.set_xlabel(r'%s' % (label_names[i]), fontsize=50, labelpad=20)
        ax0.set_ylabel(r'$\Delta \ $%s' % (label_names[i]), fontsize=50, labelpad=20)

        # Set axes limits
        ax0.tick_params(labelsize=40, width=1, length=10)
        ax0.set_ylim(resid_lims[i])
        x_min, x_max = ax0.get_xlim()
        ax0.plot([x_min,x_max], [0,0], 'k--', lw=3)
        ax0.set_xlim(x_min, x_max)
        
        # Annotate median and std of residuals
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=3)
        if i==0:
            ax0.annotate('$\widetilde{{m}}=${0:6.1f}$;\ s \ =$ \ {1:6.1f}'.format(bias[i],scatter[i],width=6), 
                         xy=(0.03, 0.82), xycoords='axes fraction', fontsize=40, bbox=bbox_props)
        else:
            ax0.annotate('$\widetilde{{m}}=${0:6.2f}$;\ s \ =$ \ {1:6.2f}'.format(bias[i],scatter[i],width=6), 
                         xy=(0.03, 0.82), xycoords='axes fraction', fontsize=40, bbox=bbox_props)

        # Set axes ticks
        start, end = ax0.get_xlim()
        stepsize = (end-start)/4
        if i==0:
            xticks = np.round(np.arange(start,end,stepsize)[1:], -2)
        else:
            xticks = np.round(np.arange(start,end,stepsize)[1:], 1)
        start, end = ax0.get_ylim()
        stepsize = (end-start)/4
        if i==0:
            yticks = np.round(np.arange(start,end,stepsize)[1:], -2)
        else:
            yticks = np.round(np.arange(start,end,stepsize)[1:], 1)
        ax0.xaxis.set_ticks(xticks)
        ax0.yaxis.set_ticks(yticks)
        
        ax1 = plt.subplot(gs[i%int((y_data.shape[1]/2)+1),x_plt_indx+1])
        
        xmin, xmax = resid_lims[i]
    
        y_a = resid_a[:,i][(resid_a[:,i]>=xmin)&(resid_a[:,i]<=xmax)]
        y_b = resid_b[:,i][(resid_b[:,i]>=xmin)&(resid_b[:,i]<=xmax)]

        a = sns.distplot(y_a, vertical=True, hist=False, rug=False, ax=ax1,
                         kde_kws={"color": points.cmap(200), "lw": 4})
        b = sns.distplot(y_b,vertical=True,hist=False, rug=False, ax=ax1,
                         kde_kws={"color": points.cmap(100), "lw": 4})

        a.set_ylim(resid_lims[i])
        b.set_ylim(resid_lims[i])

        ax1.tick_params(
        axis='x',          
        which='both',     
        bottom=False,      
        top=False,         
        labelbottom=False,width=1,length=10)

        ax1.tick_params(
        axis='y',          
        which='both',   
        left=False,     
        right=True,        
        labelleft=False,
        labelright=True,
        labelsize=40,width=1,length=10)
        ax1.xaxis.set_ticks([])
        ax1.yaxis.set_ticks(yticks)


    # Create colorbar
    cbar_ax = fig.add_axes([0.88, 0.22, 0.02, 0.6])
    fig.colorbar(points,cax=cbar_ax)
    cbar = fig.colorbar(points, cax=cbar_ax, extend='neither', 
                        spacing='proportional', orientation='vertical')
    cbar.set_label(r'$S/N$', size=70)
    cbar.ax.tick_params(labelsize=40,width=1,length=10) 
    start, end = int(np.round(np.min(snr),-1)), int(np.max(snr))
    stepsize = int(np.round((end-start)/4,-1))
    tick = end
    yticks = []
    while tick>start:
        yticks = [tick] + yticks
        tick-=stepsize
    yticks = np.array(yticks,dtype=int)
    ytick_labs = np.array(yticks,dtype=str)
    ytick_labs[-1]='$>$'+ytick_labs[-1]
    cbar.set_ticks(yticks)
    cbar_ax.set_yticklabels(ytick_labs)
    
    
    # Set x and y figure labels
    fig.text(0.06, 0.5, y_lab, ha='center', va='center', 
             rotation='vertical', fontsize=80)
    fig.text(0.5, 0.04, x_lab, ha='center', va='center',
            fontsize=80)


    fig.subplots_adjust(wspace=.1, hspace=.6)
    fig.subplots_adjust(right=0.82, left=0.15, bottom=0.07)
    if savename is not None:
        plt.savefig(savename)

    plt.show()
    
def plot_compare_estimates_resid_synth(x_data, y_data, savename=None, 
                           x_lab=r'$ASPCAP \ \ \ DR14$', 
                           y_lab=r'$(SN\ Cycle$-$GAN) - DR14$',
                           snr_max=200, cmap='Blues', snr_cutoff=100,
                           resid_lims = [[-1000., 1000.], [-2, 2], [-1, 1], 
                             [-1., 1.], [-1., 1.], [-1., 1.]]):
    plt.rcParams['axes.facecolor']='white'
    sns.set_style("ticks")
    plt.rcParams['axes.grid']=True
    plt.rcParams['grid.color']='gray'
    plt.rcParams['grid.alpha']='0.4'

    # label names
    label_names = [r'$T_{\mathrm{eff}}$',r'$\log(g)$',r'$v_{micro}$',r'$[C/H]$',
                   r'$[N/H]$',r'$[O/H]$',r'$[Na/H]$',r'$[Mg/H]$',r'$[Al/H]$',
                   r'$[Si/H]$',r'$[P/H]$',r'$[S/H]$',r'$[K/H]$',r'$[Ca/H]$',
                   r'$[Ti/H]$',r'$[V/H]$',r'[$Cr/H$]',r'$[Mn/H]$',r'$[Fe/H]$',
                   r'$[Co/H]$',r'$[Ni/H]$',r'[$Cu/H$]',r'$[Ge/H]$',r'$[C12/C13]$',r'$v_{macro}$']



    # Calculate residual, median residual, and std of residuals
    resid = y_data - x_data
    bias = np.median(resid, axis=0)
    scatter = np.std(resid, axis=0)
    
    # Plot data
    fig = plt.figure(figsize=(40, 90)) 
    gs = gridspec.GridSpec(13, 5,  width_ratios=[4., 1., 1.8, 4., 1.])
    x_plt_indx = 0
    for i in range(y_data.shape[1]):

        # Set column index
        if i>=(int((y_data.shape[1]/2)+1)):
            x_plt_indx=3
        ax0 = plt.subplot(gs[i%int((y_data.shape[1]/2)+1),x_plt_indx])
        #ax0 = axes[i%3,x_plt_indx]
        
        # Plot resid vs x coloured with snr
        points = ax0.scatter(x_data[:,i], resid[:,i], c='navy', s=20)

        # Set axes labels
        ax0.set_xlabel(r'%s' % (label_names[i]), fontsize=50, labelpad=20)
        ax0.set_ylabel(r'$\Delta \ $%s' % (label_names[i]), fontsize=50, labelpad=20)

        # Set axes limits
        ax0.tick_params(labelsize=40, width=1, length=10)
        ax0.set_ylim(resid_lims[i])
        x_min, x_max = ax0.get_xlim()
        ax0.plot([x_min,x_max], [0,0], 'k--', lw=3)
        ax0.set_xlim(x_min, x_max)
        
        # Annotate median and std of residuals
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=3)
        if i==0:
            ax0.annotate('$\widetilde{{m}}=${0:6.1f}$;\ s \ =$ \ {1:6.1f}'.format(bias[i],scatter[i],width=6), 
                         xy=(0.03, 0.82), xycoords='axes fraction', fontsize=40, bbox=bbox_props)
        else:
            ax0.annotate('$\widetilde{{m}}=${0:6.2f}$;\ s \ =$ \ {1:6.2f}'.format(bias[i],scatter[i],width=6), 
                         xy=(0.03, 0.82), xycoords='axes fraction', fontsize=40, bbox=bbox_props)

        # Set axes ticks
        start, end = ax0.get_xlim()
        stepsize = (end-start)/4
        if i==0:
            xticks = np.round(np.arange(start,end,stepsize)[1:], -2)
        else:
            xticks = np.round(np.arange(start,end,stepsize)[1:], 1)
        start, end = ax0.get_ylim()
        stepsize = (end-start)/4
        if i==0:
            yticks = np.round(np.arange(start,end,stepsize)[1:], -2)
        else:
            yticks = np.round(np.arange(start,end,stepsize)[1:], 1)
        ax0.xaxis.set_ticks(xticks)
        ax0.yaxis.set_ticks(yticks)
        
        ax1 = plt.subplot(gs[i%int((y_data.shape[1]/2)+1),x_plt_indx+1])
        
        xmin, xmax = resid_lims[i]
    
        y_a = resid[:,i][(resid[:,i]>=xmin)&(resid[:,i]<=xmax)]

        a = sns.distplot(y_a, vertical=True, hist=False, rug=False, ax=ax1,
                         kde_kws={"color": 'k', "lw": 4})

        a.set_ylim(resid_lims[i])

        ax1.tick_params(
        axis='x',          
        which='both',     
        bottom=False,      
        top=False,         
        labelbottom=False,width=1,length=10)

        ax1.tick_params(
        axis='y',          
        which='both',   
        left=False,     
        right=True,        
        labelleft=False,
        labelright=True,
        labelsize=40,width=1,length=10)
        ax1.xaxis.set_ticks([])
        ax1.yaxis.set_ticks(yticks)
    
    # Set x and y figure labels
    fig.text(0.06, 0.5, y_lab, ha='center', va='center', 
             rotation='vertical', fontsize=80)
    fig.text(0.5, 0.04, x_lab, ha='center', va='center',
            fontsize=80)


    fig.subplots_adjust(wspace=.1, hspace=.6)
    fig.subplots_adjust(right=0.82, left=0.15, bottom=0.07)
    if savename is not None:
        plt.savefig(savename)

    plt.show()

def plot_J_diff(wave_grid, J_tgt, J_dec, J_diff, ele_indices=[6,7,11,12,21]):

    # Choose an element:
    elem_labels = ['Teff', 'Logg', 'Vturb [km/s]','[C/H]', '[N/H]', 
                   '[O/H]', '[Na/H]', '[Mg/H]', '[Al/H]', '[Si/H]', 
                   '[P/H]', '[S/H]', '[K/H]', '[Ca/H]', '[Ti/H]', 
                   '[V/H]', '[Cr/H]', '[Mn/H]', '[Fe/H]', '[Co/H]', 
                   '[Ni/H]', '[Cu/H]', '[Ge/H]','C12/C13', 'Vmacro [km/s]']

    for ele_index in ele_indices:

        plt.close('all')
        print(elem_labels[ele_index])
        # Plot test results
        fig, axes = plt.subplots(3,1,figsize=(50,15), sharex=True)
        j_synth, = axes[0].plot(wave_grid, J_tgt[ele_index], c='maroon')
        j_obs, = axes[1].plot(wave_grid, J_dec[ele_index], c='navy')
        j_diff, = axes[2].plot(wave_grid, J_diff[ele_index], c='mediumvioletred')

        for i in range(3):
            axes[i].tick_params(labelsize=15)

        #axes[0].set_ylim((-0.005,0.001))
        #axes[1].set_ylim((-0.005,0.001))
        #axes[2].set_ylim((-0.001,0.001))

        axes[0].set_ylabel(r'$\partial$Flux',fontsize=22,labelpad=15)
        axes[1].set_ylabel(r'$\partial$Flux',fontsize=22,labelpad=15)
        axes[2].set_ylabel(r'$\partial$Flux',fontsize=22)
        plt.xlabel(r'Wavelength (\AA)',fontsize=22)

        axes[0].set_title(r'$\frac{\partial{\mathbf{x_{synth}}}}{\partial{y_j}}$', fontsize=20)
        axes[1].set_title(r'$\frac{\partial{\mathbf{x_{synth \rightarrow synth}}}}{\partial{y_j}}$', fontsize=20)
        axes[2].set_title(r'$\frac{\partial{\mathbf{x_{synth \rightarrow synth}}}}{\partial{y_j}}-\frac{\partial{\mathbf{x_{synth}}}}{\partial{y_j}}$', fontsize=20)

        plt.tight_layout()    
        plt.show()
    
def plot_veracity(teff, logg, fe_h, c_m, n_m, fe_min=-2.5, fe_max=0.5,
                  data_label=r'$StarNet$', savename=None,
                 label_names=['$T_{\mathrm{eff}}$',r'$\log(g)$','$[Fe/H]$',
                              r'[$\alpha/M$]',r'$[N/M]$',r'$[C/M]$']):
    
    fe_h[fe_h>fe_max]=fe_max
    fe_h[fe_h<fe_min]=fe_min
    
    # Load isochrones
    Teff1,Logg1 = np.load('data/teff_logg_pos25_5Gyr.npy')
    Teff2,Logg2 = np.load('data/teff_logg_neg25_5Gyr.npy')
    Teff3,Logg3 = np.load('data/teff_logg_neg75_5Gyr.npy')
    Teff4,Logg4 = np.load('data/teff_logg_neg175_5Gyr.npy')
    
    cNorm  = colors.Normalize(vmin=fe_min, vmax=fe_max)
    scalarMap = cmx.ScalarMappable(norm=cNorm,cmap='gist_rainbow_r')
    
    fig = plt.figure(figsize=(11.5, 15)) 
    gs = gridspec.GridSpec(2, 1,  height_ratios=[3., 1])
    
    # plot logg vs teff coloured by fe_h
    ax0 = plt.subplot(gs[0,0])
    points = ax0.scatter(teff, logg, c=fe_h, s=10, cmap='gist_rainbow_r')

    # y and x labels
    ax0.set_xlabel(data_label+' '+label_names[0], fontsize=35, labelpad=20)
    ax0.set_ylabel(data_label+' '+label_names[1], fontsize=35, labelpad=20)
    
    # Plot isochrones
    ax0.plot(Teff1,Logg1,c=scalarMap.to_rgba(0.25),linewidth=3,label= '+0.25',
             path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
    ax0.plot(Teff2,Logg2,c=scalarMap.to_rgba(-0.25),linewidth=3, label= '-0.25',
             path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
    ax0.plot(Teff3,Logg3,c=scalarMap.to_rgba(-0.75),linewidth=3, label= '-0.75',
             path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
    ax0.plot(Teff4,Logg4,c=scalarMap.to_rgba(-1.75),linewidth=3, label= '-1.75',
             path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])

    # Set axis ticks
    sns.set_style("ticks")
    ax0.tick_params(labelsize=20,width=1,length=10)  
    ax0.grid(True)
    ax0.set_ylim((-1.5,5.5))
    ax0.set_ylim(ax0.get_ylim()[::-1])
    ax0.set_xlim((2800,7200))
    ax0.set_xlim(plt.xlim()[::-1])
    ax0.yaxis.set_ticks([-0.5, 1., 2.5, 4])
    ax0.xaxis.set_ticks([3500, 4500, 5500, 6500])

    # Isochrone legend
    leg = plt.legend(borderaxespad=0.2,fontsize=25, loc='upper left',ncol=1,markerscale=2.,frameon=True)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(1.0)

    
    # Calculate [(C+N)/Fe]
    
    #Solar Constants
    #Caffau et al 2011
    AC_sol = 8.50 
    AC_sol_err = 0.06
    AN_sol = 7.86 
    AFe_sol = 7.50
    logCH_sol = AC_sol - 12.
    logNH_sol = AN_sol - 12.
    logFeH_sol = AFe_sol - 12.
    logCN_sol = np.log10(10**logCH_sol + 10**logNH_sol)

    CH = c_m + fe_h
    logCH_star = CH + logCH_sol
    logC_star = logCH_star + 12.
    NH = n_m + fe_h
    logNH_star = NH + logNH_sol
    logN_star = logNH_star + 12.
    logCN_star = np.log10(10**logCH_star + 10**logNH_star) 
    CNH = logCN_star - logCN_sol
    CNFe = CNH - fe_h

    # Plot C+N vs logg
    ax1 = plt.subplot(gs[1,0])

    x_data = logg
    y_data = CNFe
    z_data = fe_h

    # overplot high Fe/H
    order = (z_data).reshape(z_data.shape[0],).argsort()
    x_data = x_data[order]
    y_data = y_data[order]
    z_data = z_data[order]

    points = ax1.scatter(x_data, y_data, c=z_data, s=10, cmap='gist_rainbow_r')

    # Set axis ticks
    sns.set_style("ticks")
    ax1.tick_params(labelsize=20,width=1,length=5)  
    ax1.grid(True)
    ax1.set_ylim((-2,2))
    ax1.set_xlim((-3.,4.5))
    ax1.xaxis.set_ticks([-1.5, 0., 1.5, 3.])

    # Plot zero line
    ax1.plot([plt.xlim()[0], plt.xlim()[1]], [0,0], 'k--', lw=1.5)

    # Set x and y axis labels
    ax1.set_xlabel(data_label+' \ '+label_names[1], fontsize=35)
    ax1.set_ylabel(data_label+r'$ \ \ [\frac{C+N}{Fe}]$', fontsize=35)


    # Create colorbar
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(points, cax=cbar_ax, extend='neither', spacing='proportional', 
                        orientation='vertical', ticks=[-3., -2., -1., -0.])
    cbar.set_label(data_label+' '+label_names[2], size=35, labelpad=20)
    cbar.ax.tick_params(labelsize=20,width=1,length=5)
    yticks = np.linspace(fe_min, fe_max, 5)
    ytick_labs = np.array(yticks,dtype=str)
    ytick_labs[-1]='$>$'+ytick_labs[-1]
    ytick_labs[0]='$<$'+ytick_labs[0]
    cbar.set_ticks(yticks)
    cbar_ax.set_yticklabels(ytick_labs)
    
    fig.subplots_adjust(right=0.84, hspace=.3,left=0.16)

    if savename is not None:
        plt.savefig(savename)

    plt.show()
    
def plot_spec_resid_density(wave_grid, resid, labels, ylim, hist=True, kde=True,
                            dist_bins=180, hex_grid=300, bias='med',
                            bias_label='$\widetilde{{m}}$ \ ',
                            cmap="ocean_r", savename=None):
    # Some plotting variables for asthetics
    plt.rcParams['axes.facecolor']='white'
    sns.set_style("ticks")
    plt.rcParams['axes.grid']=False
    plt.rcParams['grid.color']='gray'
    plt.rcParams['grid.alpha']='0.4'
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    xs = np.repeat(wave_grid.reshape(1,wave_grid.shape[0]), len(resid[0]), axis=0)

    scatter_label='$s$ \ \ '
    mad_label = '$MAD$ \ '
    bias_resids = []
    ma_resids = []
    scatter_resids = []
    for i in range(len(resid)):
        if bias=='med':
            bias_resids.append(np.median(resid[i]))
        elif bias=='mean':
            bias_resids.append(np.mean(resid[i]))
        ma_resids.append(np.median(np.abs(resid[i])))
        scatter_resids.append(np.std(resid[i]))
        
    fig = plt.figure(figsize=(15, len(resid)*3)) 
    gs = gridspec.GridSpec(len(resid), 2,  width_ratios=[5., 1])
    for i in range(len(resid)):
        ax0 = plt.subplot(gs[i,0])
        ax1 = plt.subplot(gs[i,1])

        if i == 0:
            a = ax0.hexbin(xs, resid[i], gridsize=hex_grid, cmap=cmap,  bins='log')
            cmax = np.max(a.get_array())
        else:
            a = ax0.hexbin(xs, resid[i], gridsize=hex_grid, cmap=cmap,  bins='log', vmax=cmax)

        ax0.set_xlim(wave_grid[0], wave_grid[-1])
        ax0.tick_params(axis='y',
                        labelsize=20,width=1,length=10)
        ax0.tick_params(axis='x',          
                        which='both',     
                        bottom=False,      
                        top=False,         
                        labelbottom=False, width=1,length=10)
        ax0.set_ylabel(labels[i],
                       fontsize=25)
        ax0.set_ylim(ylim)

        sns.distplot(resid[i].flatten(), vertical=True, hist=hist, ax=ax1, kde=kde,
                 rug=False, bins=dist_bins, kde_kws={"lw": 2., "color": a.cmap(cmax/4.), "gridsize": dist_bins}, 
                 hist_kws={"color": a.cmap(cmax*0.6), "alpha":0.5})
        ax1.tick_params(axis='x',          
                        which='both',     
                        bottom=False,      
                        top=False,         
                        labelbottom=False,width=1,length=10)   
        ax1.tick_params(axis='y',          
                        which='both',   
                        left=False,     
                        right=True,        
                        labelleft=False,
                        labelright=True,
                        labelsize=20,width=1,length=10)
        ax1.set_ylim(ylim)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
        ax1.annotate(bias_label+'$=$ \ \ {0:6.4f}\n'.format(bias_resids[i], width=6)+
                     scatter_label+'$=$ \ \ {0:6.4f}\n'.format(scatter_resids[i], width=6)+
                     mad_label+'$=$ \ {0:6.4f}'.format(ma_resids[i], width=6), 
                     xy=(0.07, 0.7), xycoords='axes fraction', fontsize=14, bbox=bbox_props)


    ax0.tick_params(axis='x',
                    bottom=True,
                    labelbottom=True,
                    labelsize=20,width=1,length=10)
    ax0.set_xlabel(r'Wavelength (\AA)',fontsize=25)

    cax = fig.add_axes([0.86, 0.15, .01, 0.72])
    cb = plt.colorbar(a, cax=cax)
    cb.set_label(r'$\log{(count)}$', size=25)
    cb.ax.tick_params(labelsize=20,width=1,length=10) 
    fig.subplots_adjust(wspace=0.01, bottom=0.6*(0.5**len(resid)), right=0.78)
    
    if savename is not None:
        plt.savefig(savename)
        
    plt.show()