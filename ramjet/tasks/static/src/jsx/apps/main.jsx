'use strict';


import React from 'react';
import { Link } from 'react-router';

import { BaseComponent } from '../components/base.jsx';
import '../../sass/pages/webapps.scss';


export class App extends BaseComponent {
    render() {
        return <div id="webapps" className="container-fluid">
            <h1>WebApps Main Page</h1>
            <article>
                <p>Ramjet Webapps:</p>
                <ul>
                    <li><Link to="/webapps/">Home</Link></li>
                    <li><Link to="htts://blog.laisky.com/">Blog</Link></li>
                    <li><Link to="/webdemo/">Hello, World</Link></li>
                    <li><Link to="/twitter/login/">Login</Link></li>
                </ul>
            </article>
        </div>
    }
}
