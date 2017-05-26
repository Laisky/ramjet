'use strict';

import 'babel-polyfill';

import React from 'react';
import ReactDOM from 'react-dom';
import {
    Router,
    Route,
    Redirect,
    Link,
    IndexRedirect,
    browserHistory
} from 'react-router';
import { syncHistoryWithStore } from 'react-router-redux';

// redux
import { Provider } from 'react-redux';
import { RootReducer, store } from './reducers';

// apps
import { App } from './apps/main.jsx';


const history = syncHistoryWithStore(browserHistory, store);


ReactDOM.render(
    <Provider store={store}>
        <Router history={browserHistory}>
            <Route name="home" path="/webapps/" component={App}></Route>
            <Redirect from="*" to="/404.html"/>
        </Router>
    </Provider>, document.getElementById('body'));
