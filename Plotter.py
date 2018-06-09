from matplotlib import pyplot as pl
import numpy as np
import sys
import colorsys


class PlotterJDM:

  def __init__(self, plotTrajParams):
    self.plotTrajParams = plotTrajParams



  def plotTrajData(self, longData, longDiag, longDPS, model,
    replaceFigMode=True, yLimUseData=False,showConfInt=False, adjustBottomHeight=0.1):
    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(1, figsize=figSizeInch)
    pl.clf()
    nrRows = self.plotTrajParams['nrRows']
    nrCols = self.plotTrajParams['nrCols']

    nrBiomk = longData[0].shape[1]

    minX = np.min([np.min(dpsCurr) for dpsCurr in longDPS])
    maxX = np.max([np.max(dpsCurr) for dpsCurr in longDPS])

    xs = np.linspace(minX, maxX, 100)
    diagNrs = self.plotTrajParams['diagNrs']

    nrSubjLong = len(longData)

    dysScoresSF = model.predPopDys(xs)
    modelPredSB = model.predPop(xs)

    print('xs', xs)
    print('modelPredSB[:,0]', modelPredSB[:,0])

    lw = 3.0

    # first plot the dysfunctionality biomarkers
    ax = pl.subplot(nrRows, nrCols, 1)
    ax.set_title('dysfunc all')
    moveTicksInside(ax)
    for f in range(model.nrFuncUnits):
      ax.plot(xs, dysScoresSF[:,f], 'k-', linewidth=lw)

    # for each unit, plot all biomarkers against the dysfunctional scores
    for f in range(model.nrFuncUnits):
      ax2 = pl.subplot(nrRows, nrCols, f+2)
      ax2.set_title('dysfunc %d' % f)
      moveTicksInside(ax2)
      biomkInCurrUnit = np.where(self.plotTrajParams['mapBiomkToFuncUnits'] == f)[0]
      for b in range(len(biomkInCurrUnit)):
        ax2.plot(dysScoresSF[:,f], modelPredSB[:,biomkInCurrUnit[b]], 'k-', linewidth=lw)

      ax2.set_xlim((0,1))

    for b in range(nrBiomk):
      ax = pl.subplot(nrRows, nrCols, b + model.nrFuncUnits + 2)
      ax.set_title('biomk %d func %d' % (b, self.plotTrajParams['mapBiomkToFuncUnits'][b]))
      moveTicksInside(ax)

      pl.plot(xs, modelPredSB[:,b], 'k-', linewidth=lw)  # label='sigmoid traj %d' % b
      if showConfInt:
        pass
        # pl.fill(np.concatenate([xs, xs[::-1]]), np.concatenate([fsCurr - 1.9600 * stdDevs[b],
        #   (fsCurr + 1.9600 * stdDevs[b])[::-1]]), alpha=.3, fc='b', ec='None')
        # label='conf interval (1.96*std)')

      # print(xs[50:60], fsCurr[50:60], thetas[b,:])
      # print(asda)


      ############# spagetti plot subjects ######################
      counterDiagLegend = dict(zip(diagNrs, [0 for x in range(diagNrs.shape[0])]))
      for s in range(nrSubjLong):
        labelCurr = None
        if counterDiagLegend[longDiag[s]] == 0:
          labelCurr = self.plotTrajParams['diagLabels'][longDiag[s]]
          counterDiagLegend[longDiag[s]] += 1

        # print('longDPS', longDPS)
        # print('len(longDPS)', len(longDPS))
        # print('longDPS[s].shape', longDPS[s].shape)
        # print('longData[s][:, b]', longData[s][:, b].shape)
        pl.plot(longDPS[s], longData[s][:, b],
          c=self.plotTrajParams['diagColors'][longDiag[s]],
          label=labelCurr,alpha=0.5)

      pl.xlim(np.min(minX), np.max(maxX))

      minY = np.min([np.min(dataCurr[:,b]) for dataCurr in longData])
      maxY = np.max([np.max(dataCurr[:,b]) for dataCurr in longData])
      delta = (maxY - minY) / 10
      pl.ylim(minY - delta, maxY + delta)

    fs = 15

    fig.text(0.02, 0.6, 'Z-score of biomarker', rotation='vertical', fontsize=fs)
    fig.text(0.4, 0.052, 'disease progression score', fontsize=fs)

    # adjustCurrFig(self.plotTrajParams)
    pl.gcf().subplots_adjust(bottom=adjustBottomHeight, left=0.05, right=0.95)

    # pl.tight_layout(pad=30)
    # fig.suptitle('cluster trajectories', fontsize=20)

    h, axisLabels = ax.get_legend_handles_labels()
    # print(h[2:4], labels[2:4])
    # legend =  pl.legend(handles=h, bbox_to_anchor=self.plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])
    # legend = pl.legend(handles=h, loc='upper center', ncol=self.plotTrajParams['legendCols'])

    legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=self.plotTrajParams['legendCols'], labelspacing=0.)
    # set the linewidth of each legend object
    # for i,legobj in enumerate(legend.legendHandles):
    #   legobj.set_linewidth(4.0)
    #   legobj.set_color(self.plotTrajParams['diagColors'][diagNrs[i]])

    # mng = pl.get_current_fig_manager()
    # print(self.plotTrajParams['SubfigClustMaxWinSize'])
    # print(adsds)
    # mng.resize(*self.plotTrajParams['SubfigClustMaxWinSize'])

    if replaceFigMode:
      fig.show()
    else:
      pl.show()

    # print("Plotting results .... ")
    pl.pause(0.05)
    return fig



class PlotterGP:

  def __init__(self, plotTrajParams):
    self.plotTrajParams = plotTrajParams

  def plotTraj(self, gpModel, list_biom = [], replaceFig=True):
    nrBiomk = gpModel.N_biom
    list_biom = range(nrBiomk)
    # Plot method
    newX = np.linspace(gpModel.minX, gpModel.maxX, 30).reshape([30, 1])
    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(1, figsize = figSizeInch)
    pl.clf()

    scaledYarrayB = [gpModel.applyScalingY(gpModel.Y_array[b], b) for b in range(nrBiomk)]
    print('gpModel.X_array[b]', scaledYarrayB)
    # print(asda)
    min_yB = [np.min(scaledYarrayB[b].reshape(-1)) for b in range(nrBiomk)]
    max_yB = [np.max(scaledYarrayB[b].reshape(-1)) for b in range(nrBiomk)]
    deltaB = [(max_yB[b] - min_yB[b])/5 for b in range(nrBiomk)]

    # max_y = np.max([np.float(item) for sublist in gpModel.Y_array for item in sublist])
    # min_y = np.min([np.float(item) for sublist in gpModel.Y_array for item in sublist])

    nrRows = self.plotTrajParams['nrRows']
    nrCols = self.plotTrajParams['nrCols']

    predictedBiomksXB = gpModel.predictBiomk(newX)

    for bio_pos, b in enumerate(list_biom):
      ax = pl.subplot(nrRows, nrCols, bio_pos + 1)
      pl.title(self.plotTrajParams['labels'][b])

      # plot traj samples
      nrSamples = 500
      newXScaledX, trajSamplesXS = gpModel.sampleBiomkTrajPosterior(newX, b, nrSamples)
      for i in range(nrSamples):
        ax.plot(gpModel.applyScalingX(newX, b), gpModel.applyScalingY(trajSamplesXS[:,i], b), lw = 0.05,
        color = 'red')

      # plot subject data
      for sub in range(gpModel.N_samples):
        x_data = np.array([gpModel.X_array[b][k][0] for k in range(int(np.sum(gpModel.N_obs_per_sub[b][:sub])),
                                                                        np.sum(gpModel.N_obs_per_sub[b][:sub + 1]))])
        y_data = np.array([gpModel.Y_array[b][k][0] for k in range(int(np.sum(gpModel.N_obs_per_sub[b][:sub])),
                                                                        np.sum(gpModel.N_obs_per_sub[b][:sub + 1]))])
        # if sub in range(gpModel.N_samples)[::20]:
        #   print('x_data', x_data)
        #   print('y_data', y_data)
        #   print('gpModel.applyScalingX(x_data, b)', gpModel.applyScalingX(x_data, b))
        #   print('gpModel.applyScalingY(y_data, b)', gpModel.applyScalingY(y_data, b))
        #   print('-----------')

        ax.plot(gpModel.applyScalingX(x_data, b),
                gpModel.applyScalingY(y_data, b), color = 'green', lw = 0.5)
        ax.scatter(gpModel.applyScalingX(x_data, b),
          gpModel.applyScalingY(y_data, b), color='green', lw=0.1)

      # plot main traj
      ax.plot(gpModel.applyScalingX(newX, b),
              gpModel.applyScalingY(predictedBiomksXB[:,b], b),
              lw = 2, color = 'black')

      ax.plot(gpModel.applyScalingX(np.array([gpModel.minX, gpModel.maxX]), b), [min_yB[b], max_yB[b]], color=(0.5,0.5,0.5), lw=2)

      ax.set_ylim([min_yB[b]-deltaB[b], max_yB[b]+deltaB[b]])


      # ax.set_ylim([-0.2,1.3])
      ax.legend(loc='upper right')

    if replaceFig:
      fig.show()
    else:
      pl.show()
    pl.pause(0.05)

    return fig

  def plotCompWithTrueParams(self, gpModel, list_biom = [], replaceFig=True):

    nrBiomk = gpModel.N_biom
    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(2, figsize = figSizeInch)
    pl.clf()

    # max_y = np.max([np.float(item) for sublist in gpModel.Y_array for item in sublist])
    # min_y = np.min([np.float(item) for sublist in gpModel.Y_array for item in sublist])

    scaledYarrayB = [gpModel.applyScalingY(gpModel.Y_array[b], b) for b in range(nrBiomk)]
    min_yB = [np.min(scaledYarrayB[b].reshape(-1)) for b in range(nrBiomk)]
    max_yB = [np.max(scaledYarrayB[b].reshape(-1)) for b in range(nrBiomk)]
    deltaB = [(max_yB[b] - min_yB[b])/5 for b in range(nrBiomk)]

    nrRows = self.plotTrajParams['nrRows']
    nrCols = self.plotTrajParams['nrCols']

    ######### compare subject shifts ##########

    subShiftsTrueMarcoFormat = self.plotTrajParams['trueParams']['subShiftsTrueMarcoFormat']
    # estimShifts = gpModel.params_time_shift[0,:]
    delta = (gpModel.maxX - gpModel.minX) * 0
    newXs = np.linspace(gpModel.minX - delta, gpModel.maxX + delta, num=100).reshape([100,1])
    newXsScaled = gpModel.applyScalingX(newXs, biomk=0)
    print('gpModel.X', len(gpModel.X), len(gpModel.X[0]))
    nrSubjToSkip = 20
    xSmall = [b[::nrSubjToSkip] for b in gpModel.X]
    ySmall = [b[::nrSubjToSkip] for b in gpModel.Y]
    stagingDistSX, meanStagesS = gpModel.StageSubjects(xSmall, ySmall, newXs.reshape(-1, 1))

    meanStagesS = np.array(meanStagesS)
    print('stagingDistSX', len(stagingDistSX), stagingDistSX[0])
    print('meanStagesS', meanStagesS.shape, meanStagesS)
    print('maxLikStages', [newXsScaled[np.argmax(stagingDistSX[s])] for s in range(len(stagingDistSX))])
    print('subShiftsTrueMarcoFormat', subShiftsTrueMarcoFormat.shape, subShiftsTrueMarcoFormat[::nrSubjToSkip])
    sys.stdout.flush()
    # print(ads)
    ax = pl.subplot(nrRows, nrCols, 1)
    pl.scatter(meanStagesS, subShiftsTrueMarcoFormat[::nrSubjToSkip])
    percSubjUsed = int(100/nrSubjToSkip)
    pl.title('Subject shifts (%d %% of subj.)' % percSubjUsed)
    pl.xlabel('estimated shifts')
    pl.ylabel('true shifts')
    ax.set_ylim([np.min(subShiftsTrueMarcoFormat), np.max(subShiftsTrueMarcoFormat)])

    ######### compare all trajectories ##########
    trueXs = self.plotTrajParams['trueParams']['trueLineSpacedDPSs']
    trueTrajXB = self.plotTrajParams['trueParams']['trueTrajPredXB']
    newXTraj = np.linspace(gpModel.minX, gpModel.maxX, 30).reshape([30, 1])
    predictedBiomksXB = gpModel.predictBiomk(newXTraj)
    newXTrajScaledZeroOne = (newXTraj - np.min(newXTraj)) / (np.max(newXTraj) - np.min(newXTraj))
    trueXsScaledZeroOne = (trueXs - np.min(trueXs)) / (np.max(trueXs) - np.min(trueXs))

    yMinAll = np.min(min_yB)
    yMaxAll = np.max(max_yB)
    deltaAll = (yMaxAll - yMinAll) / 5

    if self.plotTrajParams['allTrajOverlap']:
      ax2 = pl.subplot(nrRows, nrCols, 2)
      pl.title('all trajectories')
      ax2.set_ylim([yMinAll - deltaAll, yMaxAll + deltaAll])
      for b in range(gpModel.N_biom):

        ax2.plot(newXTrajScaledZeroOne,
                 gpModel.applyScalingY(predictedBiomksXB[:,b], b), '-',lw=2
          ,c=self.plotTrajParams['colorsTraj'][b], label=self.plotTrajParams['labels'][b])
        # print('trueTrajXB[:,b]', trueTrajXB[:,b])
        ax2.plot(trueXsScaledZeroOne, trueTrajXB[:,b], '--', lw=2
          ,c=self.plotTrajParams['colorsTraj'][b])

      ax2.legend(loc='lower right',ncol=4)
      nrPlotsSoFar = 2
    else:
      ax2 = pl.subplot(nrRows, nrCols, 2)
      pl.title('all estimated trajectories')
      ax2.set_ylim([yMinAll - deltaAll, yMaxAll + deltaAll])
      for b in range(gpModel.N_biom):

        ax2.plot(newXTrajScaledZeroOne,
                 gpModel.applyScalingY(predictedBiomksXB[:,b], b), '-',lw=2
          ,c=self.plotTrajParams['colorsTraj'][b], label=self.plotTrajParams['labels'][b])

      ax2.legend(loc='lower right',ncol=4)


      ax3 = pl.subplot(nrRows, nrCols, 3)
      pl.title('all true trajectories')
      ax3.set_ylim([yMinAll - deltaAll, yMaxAll + deltaAll])
      for b in range(gpModel.N_biom):

        ax3.plot(trueXsScaledZeroOne, trueTrajXB[:,b], '--', lw=2
          ,c=self.plotTrajParams['colorsTraj'][b])

      ax3.legend(loc='lower right',ncol=4)

      nrPlotsSoFar = 3

    ######### compare biomarker trajectories one by one ##########

    for b in range(gpModel.N_biom):
      ax4 = pl.subplot(nrRows, nrCols, b+nrPlotsSoFar+1)
      pl.title(self.plotTrajParams['labels'][b])

      ax4.plot(newXTrajScaledZeroOne,
               gpModel.applyScalingY(predictedBiomksXB[:,b], b), '-',lw=2,
        c=self.plotTrajParams['colorsTraj'][b], label='estimated')


      ax4.plot(trueXsScaledZeroOne, trueTrajXB[:,b], '--', lw=2,
        c=self.plotTrajParams['colorsTraj'][b], label='true')

      ax4.set_ylim([min_yB[b] - deltaB[b], max_yB[b] + deltaB[b]])
      ax4.legend(loc='lower right')

    if replaceFig:
      fig.show()
    else:
      pl.show()
    pl.pause(0.05)

    # print(ads)
    return fig

  def Plot_predictions(self, gpModel, final_predSX, Xrange, names=[]):
    scaling = gpModel.mean_std_X[0][1] * gpModel.max_X[0]
    for i in range(len(final_predSX)):
      valid_indices = np.where(np.array(final_predSX[i]) != 0)
      print('predictions[i]', final_predSX[i].shape)
      print('Xrange', Xrange.shape)
      assert final_predSX[i].shape[0] == Xrange.shape[0]
      # print('Xrange[valid_indices]', Xrange[valid_indices])
      # print('Xrange[valid_indices] * scaling', Xrange[valid_indices] * scaling)
      # print('Xrange[valid_indices] * scaling + gpModel.mean_std_X[0][0]', Xrange[valid_indices] * scaling + gpModel.mean_std_X[0][0])
      # print('np.array(predictions[i])[valid_indices]', np.array(final_predSX[i])[valid_indices])
      pl.plot(Xrange[valid_indices] * scaling + gpModel.mean_std_X[0][0], np.array(final_predSX[i])[valid_indices])
      if len(names) > 0:
        print(np.max(final_predSX[i]))
        print(final_predSX[i])
        print(final_predSX[i] == np.max(final_predSX[i]))
        max = np.int(np.where(final_predSX[i] == np.max(final_predSX[i]))[0])
        pl.annotate(names[i], xy=(Xrange[max] * scaling + gpModel.mean_std_X[0][0], final_predSX[i][max]))
    pl.show()

def adjustCurrFig(plotTrajParams):
  fig = pl.gcf()
  # fig.set_size_inches(180/fig.dpi, 100/fig.dpi)

  mng = pl.get_current_fig_manager()
  if plotTrajParams['agg']:  # if only printing images
    pass
  else:
    maxSize = mng.window.maxsize()
    maxSize = (maxSize[0] / 2.1, maxSize[1] / 1.1)
    # print(maxSize)
    mng.resize(*maxSize)

    # mng.window.SetPosition((500, 0))
    mng.window.wm_geometry("+200+50")

  # pl.tight_layout()
  pl.gcf().subplots_adjust(bottom=0.25)

  # pl.tight_layout(pad=50, w_pad=25, h_pad=25)

def moveTicksInside(ax):
  ax.tick_params(axis='y', direction='in', pad=-30)
  ax.tick_params(axis='x', direction='in', pad=-15)