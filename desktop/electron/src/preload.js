/**
 * Picture-Aliver Electron - Preload Script
 * 
 * Exposes safe APIs to the renderer process.
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods to renderer
contextBridge.exposeInMainWorld('electronAPI', {
    // Backend URL for API calls
    getBackendUrl: () => ipcRenderer.invoke('get-backend-url'),
    
    // App information
    getAppInfo: () => ipcRenderer.invoke('get-app-info'),
    
    // Dialog methods
    openFolderDialog: (options) => ipcRenderer.invoke('open-folder-dialog', options),
    openFileDialog: (options) => ipcRenderer.invoke('open-file-dialog', options),
    saveFileDialog: (options) => ipcRenderer.invoke('save-file-dialog', options),
    
    // Notifications
    showNotification: (options) => ipcRenderer.invoke('show-notification', options),
    
    // Menu actions listener
    onMenuAction: (callback) => {
        ipcRenderer.on('menu-action', (event, action) => callback(action));
    },
    
    // Platform info
    platform: process.platform,
    
    // Version
    versions: {
        node: process.versions.node,
        chrome: process.versions.chrome,
        electron: process.versions.electron
    }
});

// Log that preload is loaded
console.log('Picture-Aliver preload script loaded');